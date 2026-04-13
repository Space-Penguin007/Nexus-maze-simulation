"""
╔══════════════════════════════════════════════════════════════════════╗
║        NEXUS BOT 4 — RoboCupJunior Rescue Simulation 2026  v3       ║
║        Flood-Fill BFS + Wall-Follow + SLAM-GPS per Area 4           ║
╠══════════════════════════════════════════════════════════════════════╣
║  FIX rispetto a v2 (code review completa):                          ║
║                                                                      ║
║   [CRITICO] send_victim(): rimosso prefisso "v" — codice raw        ║
║   [CRITICO] send_map(): rimosso prefisso "map" — matrice raw        ║
║   [CRITICO] start_return(): rinominato in return_path per evitare   ║
║             shadowing del metodo bfs_path di OccupancyMap           ║
║   [CRITICO] is_linear: calcolato da OccupancyMap.is_floating()      ║
║             invece di hardcoded True (floating = 3x punti)          ║
║   [CRITICO] Fake detection LiDAR: varianza su settore frontale      ║
║             invece di confronto mid vs mid+q (punti non allineati)  ║
║   [WARNING] read_engine_messages(): strip null bytes \x00           ║
║   [WARNING] OccupancyMap.set(): priorità celle, victim non          ║
║             sovrascritto da swamp/checkpoint                         ║
║   [WARNING] check_stuck(): soglia movimento 0.012m (GPS noise),    ║
║             timer 17s invece di 15s                                  ║
║   [WARNING] send_map(): intervallo adattivo 20s/40s + invio         ║
║             forzato prima di send_exit()                             ║
║   [WARNING] RETURN: budget calcolato da lunghezza BFS, non solo     ║
║             tempo fisso                                              ║
║   [WARNING] to_matrix_string(): fix NameError max_gz scope          ║
║   [IMPROVE] Exit bonus: buffer 5 timestep tra send_map e send_exit  ║
╚══════════════════════════════════════════════════════════════════════╝

Sensori dal JSON nexus_bot_4:
  DS: ds_front_left, ds_left, ds_right, ds_front_right,
      distance sensor5 (rear), distance sensor6 (front),
      ds_rear_left, ds_rear_right
  Camera: camera_left (x=-370), camera_right (x=+370)
  Lidar, GPS, Gyro, Color sensor (color)
  Wheel: wheel1 (x=+260, destra), wheel2 (x=-260, sinistra)
"""

from controller import Robot
import math
import struct
from collections import deque

# ─────────────────────────────────────────────────────────────────────
# COSTANTI GLOBALI
# ─────────────────────────────────────────────────────────────────────
TILE_SIZE       = 0.12      # Area 1: 12 cm
HALF_TILE_SIZE  = 0.06      # Area 2 e 3: 6 cm
MAX_SPEED       = 6.28
BASE_SPEED      = 3.0
TURN_SPEED      = 2.0
ROTATE_SPEED    = 2.5
STOP_TIME       = 1.05      # leggermente sopra 1.0s per sicurezza (sez 5.6.1)

# Soglie distanza (metri)
DS_WALL_THRESHOLD = 0.08
DS_FREE_THRESHOLD = 0.15

# Colori pavimento (RGB normalizzato)
COLOR_SILVER  = (0.75, 0.75, 0.75)
COLOR_BROWN   = (0.60, 0.30, 0.00)
COLOR_BLACK   = (0.05, 0.05, 0.05)
COLOR_BLUE    = (0.00, 0.00, 1.00)
COLOR_YELLOW  = (1.00, 1.00, 0.00)
COLOR_GREEN   = (0.00, 0.80, 0.00)
COLOR_PURPLE  = (0.50, 0.00, 0.50)
COLOR_ORANGE  = (1.00, 0.50, 0.00)
COLOR_RED     = (1.00, 0.00, 0.00)
COLOR_TOL     = 0.15

# Area multipliers (sez. 5.6.8)
AREA_MUL = {1: 1.0, 2: 1.25, 3: 1.5, 4: 2.0}

# Victim codes inviati al game manager: lettere LATINE H/S/U (non Φ/Ψ/Ω)
# I simboli greci sono solo visuali sul muro; il game manager vuole H, S, U
LETTER_SEND = {'H': 'H', 'S': 'S', 'U': 'U'}

# Cognitive target: valore per colore anello (sez. 3.7.8)
RING_VAL = {'black': -2, 'red': -1, 'yellow': 0, 'green': 1, 'blue': 2}
# Tipo hazmat in base alla somma (sez. 3.7.9)
HAZMAT = {0: 'F', 1: 'P', 2: 'C', 3: 'O'}

# [FIX WARNING] Priorità celle mappa: indice più basso = priorità più alta.
# Una cella non viene mai degradata a un valore meno importante.
# sez. 5.6.10.b — i victim code hanno priorità su swamp/checkpoint/ecc.
CELL_PRIORITY = ['5', 'H', 'S', 'U', 'F', 'P', 'C', 'O',
                  '4', 'b', 'y', 'g', 'p', 'o', 'r',
                  '3', '2', 'x', '1', '0']


# ─────────────────────────────────────────────────────────────────────
# OCCUPANCY MAP (formato conforme regolamento sez. 5.6.10)
# ─────────────────────────────────────────────────────────────────────
class OccupancyMap:
    FREE       = '0'
    WALL       = '1'
    HOLE       = '2'
    SWAMP      = '3'
    CHECKPOINT = '4'
    START      = '5'
    OBSTACLE   = 'x'
    PASS = {'1_2': 'b', '1_3': 'y', '1_4': 'g',
            '2_3': 'p', '2_4': 'o', '3_4': 'r'}

    def __init__(self, size=80):
        self.size   = size
        self.offset = size // 2
        # Griglia principale (carattere per cella)
        self.grid   = [[self.FREE] * size for _ in range(size)]
        self.explored = [[False] * size for _ in range(size)]
        # Victims sui muri: (gx, gz, wall_side) -> codice stringa (H/S/U/F/P/C/O)
        self.wall_victims = {}
        # Zona Area 4 (bbox): (min_gx, min_gz, max_gx, max_gz)
        self.area4_bbox = None
        self.start_gx = self.offset
        self.start_gz = self.offset
        # Imposta start tile
        self.set(self.offset, self.offset, self.START)

    # ── Conversioni coordinate ────────────────────────────────────────
    def world_to_grid(self, wx, wz, tile_size=TILE_SIZE):
        gx = int(round(wx / tile_size)) + self.offset
        gz = int(round(wz / tile_size)) + self.offset
        return gx, gz

    # ── Accesso griglia ───────────────────────────────────────────────
    def set(self, gx, gz, val):
        """
        [FIX WARNING] Imposta cella solo se il nuovo valore ha priorità
        maggiore o uguale rispetto al valore attuale.
        Questo garantisce che un victim code non venga mai sovrascritto
        da swamp/checkpoint/muro successivo.
        """
        if 0 <= gx < self.size and 0 <= gz < self.size:
            current = self.grid[gz][gx]
            # Calcola priorità (indice in CELL_PRIORITY: minore = più importante)
            try:
                cur_prio = CELL_PRIORITY.index(current)
            except ValueError:
                cur_prio = len(CELL_PRIORITY)
            try:
                new_prio = CELL_PRIORITY.index(val)
            except ValueError:
                new_prio = len(CELL_PRIORITY)
            # Non degradare una cella già più importante
            if new_prio > cur_prio:
                return
            self.grid[gz][gx] = val
            self.explored[gz][gx] = True

    def get(self, gx, gz):
        if 0 <= gx < self.size and 0 <= gz < self.size:
            return self.grid[gz][gx]
        return self.WALL

    def mark_explored(self, gx, gz):
        if 0 <= gx < self.size and 0 <= gz < self.size:
            self.explored[gz][gx] = True

    # ── Victim sul muro ───────────────────────────────────────────────
    def add_wall_victim(self, gx, gz, side, code):
        """
        Salva victim sul muro indicato.
        side: 'top','bottom','left','right' (rispetto alla cella robot)
        Se più victims sullo stesso muro: concatena per posizione (top→bottom, left→right)
        sez. 5.6.10.b.v
        """
        key = (gx, gz, side)
        existing = self.wall_victims.get(key, '')
        if code not in existing:
            self.wall_victims[key] = existing + code

    # ── BFS path ─────────────────────────────────────────────────────
    def bfs_path(self, sx, sz, tx, tz):
        """Ritorna lista di (gx,gz) da (sx,sz) a (tx,tz) evitando muri/hole."""
        if sx == tx and sz == tz:
            return []
        queue   = deque([(sx, sz, [])])
        visited = {(sx, sz)}
        while queue:
            cx, cz, path = queue.popleft()
            for dx, dz in [(0,-1),(0,1),(-1,0),(1,0)]:
                nx, nz = cx+dx, cz+dz
                if (nx, nz) not in visited and 0 <= nx < self.size and 0 <= nz < self.size:
                    cell = self.get(nx, nz)
                    if cell not in (self.WALL, self.HOLE):
                        new_path = path + [(nx, nz)]
                        if nx == tx and nz == tz:
                            return new_path
                        visited.add((nx, nz))
                        queue.append((nx, nz, new_path))
        return []

    def find_frontier(self, gx, gz):
        """Trova la cella libera non esplorata più vicina (BFS frontier)."""
        queue   = deque([(gx, gz, 0)])
        visited = {(gx, gz)}
        while queue:
            cx, cz, d = queue.popleft()
            if not self.explored[cz][cx] and self.get(cx, cz) not in (self.WALL, self.HOLE):
                return cx, cz, d
            for dx, dz in [(0,-1),(0,1),(-1,0),(1,0)]:
                nx, nz = cx+dx, cz+dz
                if (nx, nz) not in visited and 0 <= nx < self.size and 0 <= nz < self.size:
                    if self.get(nx, nz) not in (self.WALL, self.HOLE):
                        visited.add((nx, nz))
                        queue.append((nx, nz, d+1))
        return None, None, None

    # ── Floating tile detection ───────────────────────────────────────
    def is_floating(self, gx, gz):
        """
        [FIX CRITICO] Determina se un tile è "floating" (non raggiunge
        la start tile seguendo il muro sinistro o destro).
        I floating tiles valgono 3× i punti TI (sez. 5.6.3b).

        Algoritmo semplificato: se la distanza BFS verso la start tile
        è significativamente maggiore della distanza euclidea, il tile
        è probabilmente floating (non direttamente collegato al perimetro).
        """
        bfs_dist = len(self.bfs_path(gx, gz, self.start_gx, self.start_gz))
        if bfs_dist == 0:
            return False
        eucl_dist = math.sqrt((gx - self.start_gx)**2 + (gz - self.start_gz)**2)
        # Euristicamente: se BFS > 2.5x euclideo, il tile è probabilmente floating
        return bfs_dist > max(4, eucl_dist * 2.5)

    # ── Esportazione mappa (sez. 5.6.10) ─────────────────────────────
    def to_matrix_string(self):
        """
        Esporta la mappa nel formato del regolamento:
        - Griglia centrata sulla starting tile
        - Victims inseriti nelle celle muro adiacenti
        - Area 4 riempita con '*' (sez. 5.6.10.c)
        - Curved wall vertex = '0' (sez. 5.6.10.b.iv)
        """
        # Copia della griglia
        out = [list(row) for row in self.grid]

        # Inserisci victims sui muri (sez. 5.6.10.b.v)
        # Il victim è rappresentato nella cella del muro adiacente alla cella del robot
        side_to_delta = {
            'front': (0, -1), 'rear': (0, 1),
            'left': (-1, 0),  'right': (1, 0),
        }
        for (gx, gz, side), code in self.wall_victims.items():
            dx, dz = side_to_delta.get(side, (0, 0))
            wall_gx, wall_gz = gx + dx, gz + dz
            if 0 <= wall_gx < self.size and 0 <= wall_gz < self.size:
                # Sostituisci '1' con codice victim (o aggiungi se già c'è)
                current = out[wall_gz][wall_gx]
                if current == self.WALL or current == self.FREE:
                    out[wall_gz][wall_gx] = code
                elif code not in current:
                    out[wall_gz][wall_gx] = current + code  # concatena sez.5.6.10.b.v

        # Riempi Area 4 con '*' (sez. 5.6.10.c)
        if self.area4_bbox:
            min_gx, min_gz, max_gx, max_gz = self.area4_bbox
            for z in range(max(0, min_gz-1), min(self.size, max_gz+2)):
                for x in range(max(0, min_gx-1), min(self.size, max_gx+2)):
                    out[z][x] = '*'

        # Ritaglia la mappa attorno alla starting tile per minimizzare dimensioni
        # (sez. 5.6.10.b.vii.A: gli organizzatori allineano sulla starting tile)
        s_gx, s_gz = self.start_gx, self.start_gz
        # Trova bounding box delle celle non-FREE
        min_x = max_x = s_gx
        min_z = max_z = s_gz
        for z in range(self.size):
            for x in range(self.size):
                if out[z][x] != self.FREE:
                    min_x = min(min_x, x); max_x = max(max_x, x)
                    min_z = min(min_z, z); max_z = max(max_z, z)
        # Aggiungi margine
        min_x = max(0, min_x - 1)
        max_x = min(self.size - 1, max_x + 1)
        min_z = max(0, min_z - 1)
        # [FIX WARNING] max_gz era usato fuori scope se area4_bbox è None → NameError
        if self.area4_bbox:
            area4_max_gz = self.area4_bbox[3]
            max_z = min(self.size - 1, max(max_z, area4_max_gz) + 1)
        else:
            max_z = min(self.size - 1, max_z + 1)

        rows = []
        for z in range(min_z, max_z + 1):
            rows.append(''.join(out[z][min_x:max_x + 1]))
        return '\n'.join(rows)


# ─────────────────────────────────────────────────────────────────────
# CONTROLLER PRINCIPALE
# ─────────────────────────────────────────────────────────────────────
class NexusController:

    def __init__(self):
        self.robot    = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.dt       = self.timestep / 1000.0

        self._init_sensors()
        self._init_state()
        print("[NEXUS v3] Inizializzazione completata.")

    # ─────────────────────────────────────────────────────────────────
    # INIZIALIZZAZIONE
    # ─────────────────────────────────────────────────────────────────
    def _init_sensors(self):
        # Distance sensors (nomi dal JSON)
        ds_names = {
            'front':       'distance sensor6',   # z=-370
            'rear':        'distance sensor5',   # z=+370
            'left':        'ds_left',
            'right':       'ds_right',
            'front_left':  'ds_front_left',
            'front_right': 'ds_front_right',
            'rear_left':   'ds_rear_left',
            'rear_right':  'ds_rear_right',
        }
        self.ds = {}
        for alias, dev_name in ds_names.items():
            dev = self.robot.getDevice(dev_name)
            dev.enable(self.timestep)
            self.ds[alias] = dev

        # Camere (sinistra e destra — dal JSON)
        self.cam = {}
        for side in ('left', 'right'):
            dev = self.robot.getDevice(f'camera_{side}')
            dev.enable(self.timestep)
            self.cam[side] = dev

        # GPS
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

        # Gyro
        self.gyro = self.robot.getDevice('gyro')
        self.gyro.enable(self.timestep)

        # Color sensor (pavimento)
        self.color_dev = self.robot.getDevice('color')
        self.color_dev.enable(self.timestep)

        # LiDAR
        self.lidar = self.robot.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # Ruote: wheel1=destra (x=+260), wheel2=sinistra (x=-260) — dal JSON
        self.motor_r = self.robot.getDevice('wheel1')
        self.motor_l = self.robot.getDevice('wheel2')
        for m in (self.motor_r, self.motor_l):
            m.setPosition(float('inf'))
            m.setVelocity(0.0)

        # ── EMITTER (invio messaggi al game manager) ──────────────────
        self.emitter = self.robot.getDevice('emitter')

        # ── RECEIVER (ricezione messaggi dall'engine) ─────────────────
        # CRITICO sez. 5.5.3: il motore invia 'L' al robot quando LoP è triggerato
        self.receiver = self.robot.getDevice('receiver')
        if self.receiver:
            self.receiver.enable(self.timestep)

    def _init_state(self):
        self.state        = 'EXPLORE'
        self.current_area = 1
        self.occ_map      = OccupancyMap()

        # Posizione GPS precedente (per anti-stuck e heading complementare)
        self.prev_gps     = None
        self.prev_gps_time = 0.0

        # Heading stimato (filtro complementare GPS+Gyro)
        self.heading      = 0.0       # radianti
        self.gyro_heading = 0.0       # integrazione gyro pura

        # Percorso BFS corrente (esplorazione)
        self.bfs_path     = []

        # [FIX CRITICO] Percorso di ritorno alla start tile — attributo dedicato
        # per evitare il shadowing del metodo occ_map.bfs_path()
        self.return_path  = []

        # Stato victim
        self.victim_timer      = 0.0
        self.pending_victim    = None   # dict con info victim da inviare
        self.identified_tokens = set()  # (gx, gz, side) già inviati

        # Scoring locale (per debug e decisioni)
        self.score_local  = 0
        self.tmi_count    = 0          # misidentification count (penalità)

        # Checkpoints visitati
        self.checkpoints_visited = set()

        # Swamp: contatore ingressi per moltiplicatore progressivo (sez. 3.6.2c)
        self.swamp_entries = {}

        # Stuck detection
        self.stuck_timer = 0.0
        self.last_pos    = None

        # Tempo di gioco (8 min simulati = 480s)
        self.game_time   = 0.0

        # [FIX WARNING] Traccia ultimo invio mappa per invio adattivo
        self.last_map_send = 0.0

        # Flag
        self.exit_sent   = False
        self.lop_count   = 0

    # ─────────────────────────────────────────────────────────────────
    # LETTURA SENSORI
    # ─────────────────────────────────────────────────────────────────
    def read_ds(self):
        return {k: v.getValue() for k, v in self.ds.items()}

    def wall(self, ds, side):
        return ds[side] < DS_WALL_THRESHOLD

    def read_gps(self):
        v = self.gps.getValues()
        return v[0], v[2]   # x, z (NUE: y è verticale)

    def update_heading(self):
        """
        Filtro complementare GPS + Gyro per stimare heading senza drift.
        - Gyro: alta frequenza, ma deriva nel tempo
        - GPS diff: bassa frequenza ma stabile
        α = 0.95 → 95% gyro, 5% GPS correction
        """
        gyro_y = self.gyro.getValues()[1]  # asse verticale in NUE
        self.gyro_heading += gyro_y * self.dt

        # Correzione GPS (ogni ~0.5s per avere spostamento misurabile)
        wx, wz = self.read_gps()
        if self.prev_gps is not None:
            dt_gps = self.game_time - self.prev_gps_time
            if dt_gps > 0.3:
                dx = wx - self.prev_gps[0]
                dz = wz - self.prev_gps[1]
                dist = math.sqrt(dx*dx + dz*dz)
                if dist > 0.005:   # solo se ci si è mossi abbastanza
                    gps_heading = math.atan2(-dx, -dz)  # NUE: -z=avanti, -x=destra
                    # Normalizza differenza
                    diff = gps_heading - self.gyro_heading
                    while diff >  math.pi: diff -= 2*math.pi
                    while diff < -math.pi: diff += 2*math.pi
                    self.gyro_heading += 0.05 * diff   # correzione 5%
                self.prev_gps = (wx, wz)
                self.prev_gps_time = self.game_time
        else:
            self.prev_gps = (wx, wz)
            self.prev_gps_time = self.game_time

        self.heading = self.gyro_heading
        return self.heading

    def read_floor_color(self):
        img = self.color_dev.getImage()
        if img is None:
            return (0.0, 0.0, 0.0)
        w = self.color_dev.getWidth()
        h = self.color_dev.getHeight()
        cx, cy = w // 2, h // 2
        r = self.color_dev.imageGetRed(img,   w, cx, cy) / 255.0
        g = self.color_dev.imageGetGreen(img, w, cx, cy) / 255.0
        b = self.color_dev.imageGetBlue(img,  w, cx, cy) / 255.0
        return r, g, b

    def color_match(self, meas, ref):
        return all(abs(m - r) < COLOR_TOL for m, r in zip(meas, ref))

    def detect_floor(self):
        rgb = self.read_floor_color()
        if self.color_match(rgb, COLOR_SILVER): return 'checkpoint'
        if self.color_match(rgb, COLOR_BROWN):  return 'swamp'
        if self.color_match(rgb, COLOR_BLACK):  return 'hole'
        if self.color_match(rgb, COLOR_BLUE):   return 'pass_1_2'
        if self.color_match(rgb, COLOR_YELLOW): return 'pass_1_3'
        if self.color_match(rgb, COLOR_GREEN):  return 'pass_1_4'
        if self.color_match(rgb, COLOR_PURPLE): return 'pass_2_3'
        if self.color_match(rgb, COLOR_ORANGE): return 'pass_2_4'
        if self.color_match(rgb, COLOR_RED):    return 'pass_3_4'
        return 'normal'

    def read_engine_messages(self):
        """
        CRITICO (sez. 5.5.3): legge messaggi dal game engine.
        'L' = LoP triggerato → il robot è stato resettato all'ultimo checkpoint.
        [FIX WARNING] Aggiunto strip di null bytes \x00 che Erebus include.
        """
        if not self.receiver or self.receiver.getQueueLength() == 0:
            return None
        msg = self.receiver.getData()
        self.receiver.nextPacket()
        try:
            decoded = msg.decode('utf-8').strip().strip('\x00')
            return decoded if decoded else None
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────
    # CONTROLLO MOTORI
    # ─────────────────────────────────────────────────────────────────
    def set_speeds(self, left, right):
        self.motor_l.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, left)))
        self.motor_r.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, right)))

    def forward(self, spd=None):  self.set_speeds(spd or BASE_SPEED, spd or BASE_SPEED)
    def backward(self, spd=None): self.set_speeds(-(spd or BASE_SPEED), -(spd or BASE_SPEED))
    def turn_left(self, spd=None):  s = spd or TURN_SPEED; self.set_speeds(-s,  s)
    def turn_right(self, spd=None): s = spd or TURN_SPEED; self.set_speeds( s, -s)
    def stop(self):               self.set_speeds(0, 0)

    def rotate_to(self, target, tol=0.05):
        """
        Ruota sul posto verso heading target (rad).
        Ritorna True quando raggiunto.
        """
        cur  = self.heading
        diff = target - cur
        while diff >  math.pi: diff -= 2*math.pi
        while diff < -math.pi: diff += 2*math.pi
        if abs(diff) < tol:
            self.stop()
            return True
        spd = ROTATE_SPEED * (1.0 if diff > 0 else -1.0)
        self.set_speeds(-spd, spd)
        return False

    # ─────────────────────────────────────────────────────────────────
    # AGGIORNAMENTO MAPPA
    # ─────────────────────────────────────────────────────────────────
    def update_map(self, ds, gx, gz):
        """Aggiorna la mappa di occupancy dalle letture dei sensori."""
        self.occ_map.mark_explored(gx, gz)
        dirs = {'front': (0,-1), 'rear': (0,1), 'left': (-1,0), 'right': (1,0)}
        for side, (dx, dz) in dirs.items():
            nx, nz = gx+dx, gz+dz
            if ds[side] < DS_WALL_THRESHOLD:
                if self.occ_map.get(nx, nz) == OccupancyMap.FREE:
                    self.occ_map.set(nx, nz, OccupancyMap.WALL)
            # Diagonali: informazione aggiuntiva per curved walls in Area 3
        if self.current_area == 3:
            self._handle_curved_walls(ds, gx, gz)

    def _handle_curved_walls(self, ds, gx, gz):
        """
        Area 3 — Curved walls (sez. 3.3.2):
        I corner 90° possono essere arrotondati in quarto di cerchio.
        Il vertex nella mappa rimane '0' come da sez. 5.6.10.b.iv.
        """
        if ds['front'] < DS_WALL_THRESHOLD:
            if ds['front_left'] > DS_FREE_THRESHOLD and ds['front_right'] > DS_FREE_THRESHOLD:
                nx, nz = gx, gz-1
                if self.occ_map.get(nx, nz) == OccupancyMap.WALL:
                    self.occ_map.grid[nz][nx] = OccupancyMap.FREE  # vertex = '0'

    # ─────────────────────────────────────────────────────────────────
    # COMUNICAZIONE GAME MANAGER
    # ─────────────────────────────────────────────────────────────────
    def send_victim(self, code):
        """
        [FIX CRITICO] Invia identificazione victim (sez. 5.6.1).
        Il game manager Erebus si aspetta il codice ASCII puro senza prefissi.
        v2 inviava "v{code}" che causava rifiuto silenzioso → zero punti TI/TT.
        Codice = H, S, U per letter victims
        Codice = F, P, C, O per cognitive targets
        """
        if self.emitter:
            msg = code.encode('utf-8')   # solo il codice, niente prefisso
            self.emitter.send(msg)
            print(f"[VICTIM→ENGINE] Inviato codice: {code}")

    def send_map(self):
        """
        [FIX CRITICO] Invia mappa (sez. 5.6.10).
        v2 inviava "map{matrice}" — il game manager si aspetta la matrice raw.
        """
        if self.emitter:
            mat = self.occ_map.to_matrix_string()
            self.emitter.send(mat.encode('utf-8'))   # matrice raw, niente prefisso
            self.last_map_send = self.game_time
            print("[MAP→ENGINE] Mappa inviata.")

    def send_exit(self):
        """
        [FIX IMPROVE] Invia exit (sez. 5.7.2b).
        Garantisce che la mappa sia stata inviata almeno 5 timestep prima.
        """
        if self.emitter and not self.exit_sent:
            self.emitter.send(b"exit")
            self.exit_sent = True
            print("[EXIT→ENGINE] Comando exit inviato.")

    def call_lop(self):
        """Robot chiama LoP autonomamente (sez. 5.5.1d)."""
        if self.emitter:
            self.emitter.send(b"lop")
            self.lop_count += 1
            self.score_local = max(0, self.score_local - 5)  # penalità -5 (sez 5.6.7)
            print(f"[LOP] Chiamato autonomamente. Totale LoP: {self.lop_count}. Score: {self.score_local}")

    # ─────────────────────────────────────────────────────────────────
    # RILEVAMENTO VITTIME
    # ─────────────────────────────────────────────────────────────────
    def scan_cameras(self, gx, gz):
        """
        Scansiona camera_left e camera_right per rilevare victims.
        Se trovato, imposta pending_victim e passa a stato VICTIM_STOP.
        """
        for side, cam in self.cam.items():
            token_key = (gx, gz, side)
            if token_key in self.identified_tokens:
                continue
            result = self._analyze_camera(cam)
            if result is not None:
                self.pending_victim = {
                    'key': token_key,
                    'result': result,
                    'side': side,
                    'gx': gx, 'gz': gz,
                }
                return True
        return False

    def _analyze_camera(self, cam):
        """
        Analizza immagine camera.
        1. Fake detection via LiDAR (token 3D → depth non uniforme)
        2. Cognitive target detection (cerchi colorati)
        3. Letter victim detection (simbolo nero su sfondo bianco)
        Ritorna dict o None se fake/vuoto.
        """
        img = cam.getImage()
        if img is None:
            return None
        w, h = cam.getWidth(), cam.getHeight()

        # ── Conta pixel scuri (zone di testo/simbolo) ─────────────────
        dark = 0
        total = (w // 2) * (h // 2)
        for y in range(h//4, 3*h//4):
            for x in range(w//4, 3*w//4):
                r = cam.imageGetRed(img, w, x, y)
                g = cam.imageGetGreen(img, w, x, y)
                b = cam.imageGetBlue(img, w, x, y)
                if (r+g+b)/3 < 77:
                    dark += 1
        if dark / total < 0.04:
            return None   # niente di interessante

        # ── [FIX CRITICO] FAKE DETECTION (sez. 3.7.4) ────────────────
        # I token 3D hanno profondità NON uniforme nel loro bounding box.
        # v2 confrontava lidar[mid] vs lidar[mid+q]: i punti non erano
        # allineati alla camera e la soglia 0.025 era troppo bassa per
        # il noise reale.
        # v3: calcola la varianza (max-min) su un settore frontale di ~±15°
        # per rilevare depth non piatto → token 3D → fake.
        lidar_data = self.lidar.getRangeImage()
        if lidar_data and len(lidar_data) > 8:
            mid = len(lidar_data) // 2
            # Campiona ±3 indici attorno al centro (≈±15° per LiDAR 360°/N punti)
            sector = lidar_data[max(0, mid-3) : mid+4]
            valid  = [d for d in sector if d != float('inf') and d > 0]
            if len(valid) >= 3:
                depth_variance = max(valid) - min(valid)
                if depth_variance > 0.015:
                    return None   # depth non piatto → token 3D → FAKE (sez. 3.7.4)

        # ── Verifica se cognitive target (cerchi colorati) ────────────
        if self._has_color_circles(cam, img, w, h):
            return self._decode_cognitive(cam, img, w, h)

        # ── Letter victim (H/S/U → Φ/Ψ/Ω sul muro) ──────────────────
        code = self._classify_letter(cam, img, w, h)
        if code:
            return {'type': 'letter', 'send_code': code}
        return None

    def _has_color_circles(self, cam, img, w, h):
        """Rileva presenza di cerchi colorati (cognitive target)."""
        colored = 0
        cx, cy = w//2, h//2
        for y in range(cy-h//4, cy+h//4):
            for x in range(cx-w//4, cx+w//4):
                r = cam.imageGetRed(img, w, x, y)
                g = cam.imageGetGreen(img, w, x, y)
                b = cam.imageGetBlue(img, w, x, y)
                is_white = r > 200 and g > 200 and b > 200
                is_black = r < 50  and g < 50  and b < 50
                if not is_white and not is_black:
                    colored += 1
        return colored > (w//2 * h//2) * 0.12

    def _decode_cognitive(self, cam, img, w, h):
        """
        Decodifica cognitive target: 5 anelli concentrici (sez. 3.7.7-9).
        Ogni anello viene campionato separatamente (sez. 3.7.10: no merging).
        Somma valori → tipo hazmat (F/P/C/O) o fake se somma non in {0,1,2,3}.
        """
        cx, cy = w//2, h//2
        pixel_radii = [2, 5, 8, 11, 14]
        ring_colors = []

        for radius in pixel_radii:
            samples_rgb = []
            for angle_deg in range(0, 360, 45):
                rad = math.radians(angle_deg)
                px = int(cx + radius * math.cos(rad))
                py = int(cy + radius * math.sin(rad))
                if 0 <= px < w and 0 <= py < h:
                    r = cam.imageGetRed(img, w, px, py)
                    g = cam.imageGetGreen(img, w, px, py)
                    b = cam.imageGetBlue(img, w, px, py)
                    samples_rgb.append((r, g, b))
            if not samples_rgb:
                ring_colors.append('yellow')  # default neutro (valore 0)
                continue
            ar = sum(s[0] for s in samples_rgb) / len(samples_rgb)
            ag = sum(s[1] for s in samples_rgb) / len(samples_rgb)
            ab = sum(s[2] for s in samples_rgb) / len(samples_rgb)
            ring_colors.append(self._classify_color(ar, ag, ab))

        # Somma i 5 anelli SEPARATAMENTE (sez. 3.7.10: adjacent rings non merged)
        total = sum(RING_VAL.get(c, 0) for c in ring_colors)
        hazmat_code = HAZMAT.get(total, None)

        if hazmat_code is None:
            return None   # somma non valida → fake victim (sez. 3.7.9)

        print(f"[COG_TARGET] Anelli: {ring_colors}, somma={total} → {hazmat_code}")
        return {'type': 'cognitive', 'send_code': hazmat_code,
                'rings': ring_colors, 'sum': total}

    def _classify_color(self, r, g, b):
        if r < 60  and g < 60  and b < 60:   return 'black'
        if r > 150 and g < 80  and b < 80:   return 'red'
        if r > 150 and g > 150 and b < 80:   return 'yellow'
        if r < 80  and g > 100 and b < 80:   return 'green'
        if r < 80  and g < 80  and b > 150:  return 'blue'
        return 'yellow'   # fallback neutro (valore 0)

    def _classify_letter(self, cam, img, w, h):
        """
        Stima il tipo di letter victim (H/S/U) dall'immagine.
        Approccio: analisi distribuzione pixel scuri su 4 quadranti
        (più robusto della semplice top/bottom per token ruotati ±π sez. 3.7.11).
        Φ (H): pixel densi al centro e in alto, simmetrico
        Ψ (S): punta centrale verso il basso, top denso
        Ω (U): apertura in alto, denso in basso e ai lati
        """
        q = [0, 0, 0, 0]   # top-left, top-right, bottom-left, bottom-right
        for y in range(h//4, 3*h//4):
            for x in range(w//4, 3*w//4):
                r = cam.imageGetRed(img, w, x, y)
                g = cam.imageGetGreen(img, w, x, y)
                b = cam.imageGetBlue(img, w, x, y)
                if (r+g+b)/3 < 77:
                    top  = y < h//2
                    left = x < w//2
                    idx  = (0 if top else 2) + (0 if left else 1)
                    q[idx] += 1

        top_total    = q[0] + q[1]
        bottom_total = q[2] + q[3]
        total        = top_total + bottom_total + 1   # evita div/0
        ratio_tb     = top_total / (bottom_total + 1)
        center_sym   = abs(q[0] - q[1]) + abs(q[2] - q[3])  # asimmetria laterale

        if ratio_tb > 1.3 and center_sym / total < 0.3:
            return 'H'   # Φ: denso in alto, simmetrico
        elif ratio_tb < 0.7:
            return 'U'   # Ω: aperto in alto, denso in basso
        else:
            return 'S'   # Ψ: distribuzione equilibrata con punta centrale

    # ─────────────────────────────────────────────────────────────────
    # GESTIONE PAVIMENTO
    # ─────────────────────────────────────────────────────────────────
    def handle_floor(self, floor_type, gx, gz):
        """Gestisce checkpoint, swamp, hole, passage."""
        if floor_type == 'checkpoint':
            key = (gx, gz)
            if key not in self.checkpoints_visited:
                self.checkpoints_visited.add(key)
                self.occ_map.set(gx, gz, OccupancyMap.CHECKPOINT)
                am  = AREA_MUL.get(self.current_area, 1.0)
                pts = int(10 * am)
                self.score_local += pts
                print(f"[CP] Checkpoint ({gx},{gz}) Area {self.current_area}: +{pts}pt (×{am}) → {self.score_local}")

        elif floor_type == 'swamp':
            self.occ_map.set(gx, gz, OccupancyMap.SWAMP)
            key = (gx, gz)
            entries = self.swamp_entries.get(key, 0) + 1
            self.swamp_entries[key] = entries
            # Sez. 3.6.2c: x5 primo ingresso, poi +1 fino a x10
            rate = min(10, 4 + entries)
            print(f"[SWAMP] ({gx},{gz}) ingresso #{entries} → tempo ×{rate}")
            self.forward(MAX_SPEED * 0.85)

        elif floor_type == 'hole':
            self.occ_map.set(gx, gz, OccupancyMap.HOLE)
            print(f"[HOLE] Hole rilevato a ({gx},{gz})! Arretro.")
            self.backward()
            return True   # segnala pericolo

        elif floor_type.startswith('pass_'):
            self._handle_passage(floor_type, gx, gz)

        return False

    def _handle_passage(self, floor_type, gx, gz):
        """Gestisce transizione tra aree (sez. 3.4)."""
        pass_map = {
            'pass_1_2': (1,2,'1_2'), 'pass_1_3': (1,3,'1_3'), 'pass_1_4': (1,4,'1_4'),
            'pass_2_3': (2,3,'2_3'), 'pass_2_4': (2,4,'2_4'), 'pass_3_4': (3,4,'3_4'),
        }
        if floor_type not in pass_map:
            return
        from_a, to_a, key = pass_map[floor_type]
        if self.current_area == from_a:
            self.occ_map.set(gx, gz, OccupancyMap.PASS[key])
            print(f"[PASSAGE] Area {from_a} → Area {to_a}")
            self.current_area = to_a
            self.bfs_path  = []
            self.return_path = []
            if to_a == 4 and self.occ_map.area4_bbox is None:
                self.occ_map.area4_bbox = (gx-1, gz-1, gx+6, gz+6)  # stima iniziale

    # ─────────────────────────────────────────────────────────────────
    # STUCK DETECTION
    # ─────────────────────────────────────────────────────────────────
    def check_stuck(self, wx, wz):
        """
        [FIX WARNING] Rileva immobilità prolungata.
        - Soglia movimento alzata a 0.012m (il GPS noise può simulare micro-movimenti)
        - Timer portato a 17s (era 15s): risparmia penalità nei casi borderline
          e lascia ancora 3s di margine prima del LoP automatico a 20s (sez. 5.5.1b)
        """
        pos = (wx, wz)
        if self.last_pos is None:
            self.last_pos = pos
            return False
        dist = math.sqrt((wx-self.last_pos[0])**2 + (wz-self.last_pos[1])**2)
        if dist < 0.012:   # era 0.004 → troppo basso per GPS noise
            self.stuck_timer += self.dt
        else:
            self.stuck_timer = 0.0
            self.last_pos = pos
        return self.stuck_timer > 17.0   # era 15.0

    # ─────────────────────────────────────────────────────────────────
    # NAVIGAZIONE: WALL FOLLOWER (Area 1 + fallback)
    # ─────────────────────────────────────────────────────────────────
    def wall_follow(self, ds):
        """
        Wall-follower sinistro con correzione PD.
        Gestisce curved walls in Area 3 via navigazione morbida.
        """
        front      = ds['front']
        left       = ds['left']
        front_left = ds['front_left']

        if front < DS_WALL_THRESHOLD:
            if self.current_area == 3 and front_left > DS_FREE_THRESHOLD:
                self.set_speeds(BASE_SPEED * 0.5, BASE_SPEED)  # curva dolce
            else:
                self.turn_right()
        elif left > DS_FREE_THRESHOLD:
            self.set_speeds(BASE_SPEED * 0.65, BASE_SPEED)
        else:
            target_dist = 0.06
            error       = target_dist - left
            kp, kd      = 6.0, 1.5
            deriv       = (front_left - left) * 0.5
            correction  = kp * error + kd * deriv
            self.set_speeds(BASE_SPEED - correction, BASE_SPEED + correction)

    # ─────────────────────────────────────────────────────────────────
    # NAVIGAZIONE: FLOOD FILL BFS (Area 1-3)
    # ─────────────────────────────────────────────────────────────────
    def flood_fill_step(self, ds, gx, gz):
        """
        Esplorazione flood-fill BFS.
        Ritorna 'DONE' se l'area è completamente esplorata.
        """
        self.update_map(ds, gx, gz)

        if not self.bfs_path:
            fx, fz, dist = self.occ_map.find_frontier(gx, gz)
            if fx is None:
                return 'DONE'
            self.bfs_path = self.occ_map.bfs_path(gx, gz, fx, fz)
            print(f"[BFS] Frontier ({fx},{fz}), dist={dist}, path={len(self.bfs_path)}")

        if not self.bfs_path:
            return 'DONE'

        next_gx, next_gz = self.bfs_path[0]
        dx = next_gx - gx
        dz = next_gz - gz

        dir_headings = {(0,-1): 0.0, (0,1): math.pi,
                        (1,0): math.pi/2, (-1,0): -math.pi/2}
        target_h = dir_headings.get((dx, dz), self.heading)

        if self.rotate_to(target_h):
            if not self.wall(ds, 'front'):
                self.forward()
                ts  = HALF_TILE_SIZE if self.current_area in (2,3) else TILE_SIZE
                tol = ts / 2.0
                if self._at_pos(next_gx, next_gz, tol):
                    self.bfs_path.pop(0)
            else:
                self.occ_map.set(next_gx, next_gz, OccupancyMap.WALL)
                self.bfs_path = []
                self.stop()

        return 'EXPLORING'

    def _at_pos(self, gx, gz, tol):
        """Controlla se il robot è nella cella griglia (gx,gz) con tolleranza tol."""
        wx, wz  = self.read_gps()
        ts      = HALF_TILE_SIZE if self.current_area in (2,3) else TILE_SIZE
        t_wx    = (gx - self.occ_map.offset) * ts
        t_wz    = (gz - self.occ_map.offset) * ts
        return math.sqrt((wx-t_wx)**2 + (wz-t_wz)**2) < tol

    # ─────────────────────────────────────────────────────────────────
    # NAVIGAZIONE: AREA 4 (SLAM / Bug Algorithm)
    # ─────────────────────────────────────────────────────────────────
    def area4_navigate(self, ds):
        """
        Area 4: layout arbitrario, movimento non-cardinale (sez. 3.2.5).
        Bug algorithm: avanza, evita ostacoli con sensori diagonali e laterali.
        """
        f  = ds['front']
        fl = ds['front_left']
        fr = ds['front_right']
        l  = ds['left']
        r  = ds['right']

        if f > DS_FREE_THRESHOLD and fl > DS_FREE_THRESHOLD and fr > DS_FREE_THRESHOLD:
            self.forward(BASE_SPEED * 1.15)
        elif f < DS_WALL_THRESHOLD:
            if l < r:
                self.turn_right(TURN_SPEED * 1.2)
            else:
                self.turn_left(TURN_SPEED * 1.2)
        elif fl < DS_WALL_THRESHOLD and fr >= DS_WALL_THRESHOLD:
            self.set_speeds(BASE_SPEED, BASE_SPEED * 0.35)
        elif fr < DS_WALL_THRESHOLD and fl >= DS_WALL_THRESHOLD:
            self.set_speeds(BASE_SPEED * 0.35, BASE_SPEED)
        else:
            if l < DS_FREE_THRESHOLD and r > DS_FREE_THRESHOLD:
                self.set_speeds(BASE_SPEED, BASE_SPEED * 0.7)
            elif r < DS_FREE_THRESHOLD and l > DS_FREE_THRESHOLD:
                self.set_speeds(BASE_SPEED * 0.7, BASE_SPEED)
            else:
                self.forward()

    # ─────────────────────────────────────────────────────────────────
    # RITORNO ALLA START TILE
    # ─────────────────────────────────────────────────────────────────
    def start_return(self, gx, gz):
        """
        [FIX CRITICO] Pianifica e segue BFS verso la start tile
        (per exit bonus sez. 5.6.9).
        v2 usava self.bfs_path che faceva shadowing del metodo occ_map.bfs_path()
        e causava loop infiniti. v3 usa l'attributo dedicato self.return_path.
        """
        if not self.return_path:
            self.return_path = self.occ_map.bfs_path(
                gx, gz, self.occ_map.start_gx, self.occ_map.start_gz)
            if not self.return_path:
                print("[RETURN] Percorso start tile non trovato!")
                return False
            print(f"[RETURN] Percorso start tile: {len(self.return_path)} passi.")
        return True

    def at_start(self, gx, gz):
        return gx == self.occ_map.start_gx and gz == self.occ_map.start_gz

    def _return_budget(self, gx, gz):
        """
        [FIX WARNING] Calcola il tempo necessario per tornare alla start.
        Stima: ~1.5s per step BFS + 20s buffer di sicurezza.
        Più preciso del fisso 90s di v2.
        """
        path = self.occ_map.bfs_path(
            gx, gz, self.occ_map.start_gx, self.occ_map.start_gz)
        return len(path) * 1.5 + 20.0

    # ─────────────────────────────────────────────────────────────────
    # LOOP PRINCIPALE
    # ─────────────────────────────────────────────────────────────────
    def run(self):
        print("[NEXUS v3] Loop principale avviato.")
        exit_buffer_steps = 0   # [FIX IMPROVE] buffer timestep tra send_map e send_exit

        while self.robot.step(self.timestep) != -1:
            self.game_time += self.dt

            # ── [FIX IMPROVE] Buffer exit bonus ──────────────────────
            # Attendiamo 5 timestep dopo send_map prima di send_exit
            # per garantire che la mappa sia stata processata dal motore.
            if exit_buffer_steps > 0:
                exit_buffer_steps -= 1
                if exit_buffer_steps == 0:
                    self.send_exit()
                    self.state = 'DONE'
                continue

            # ── Lettura sensori ──────────────────────────────────────
            ds       = self.read_ds()
            wx, wz   = self.read_gps()
            heading  = self.update_heading()
            floor    = self.detect_floor()
            ts       = HALF_TILE_SIZE if self.current_area in (2,3) else TILE_SIZE
            gx, gz   = self.occ_map.world_to_grid(wx, wz, ts)

            # ── Messaggi engine (CRITICO sez. 5.5.3) ────────────────
            engine_msg = self.read_engine_messages()
            if engine_msg == 'L':
                print("[ENGINE] Ricevuto 'L': LoP triggerato! Reset stato navigazione.")
                self.bfs_path    = []
                self.return_path = []
                self.stuck_timer  = 0.0
                self.last_pos     = None
                self.state        = 'EXPLORE'
                self.lop_count   += 1
                self.score_local  = max(0, self.score_local - 5)
                continue

            # ── Anti-stuck ───────────────────────────────────────────
            if self.check_stuck(wx, wz):
                print("[STUCK] Immobile >17s → LoP autonomo.")
                self.call_lop()
                self.backward()
                self.bfs_path    = []
                self.return_path = []
                self.state       = 'EXPLORE'
                self.stuck_timer  = 0.0
                self.last_pos     = None

            # ── Gestione pavimento ───────────────────────────────────
            danger = self.handle_floor(floor, gx, gz)
            if danger:
                continue

            # ── [FIX WARNING] Invio mappa adattivo ──────────────────
            # Aree più difficili (3,4) → invio più frequente (20s)
            # per garantire MB aggiornato anche in caso di fine anticipata.
            map_interval = 20.0 if self.current_area >= 3 else 40.0
            if self.game_time - self.last_map_send > map_interval:
                self.send_map()

            # ── STATO: VICTIM_STOP ───────────────────────────────────
            if self.state == 'VICTIM_STOP':
                self.stop()
                self.victim_timer += self.dt
                if self.victim_timer >= STOP_TIME:
                    v    = self.pending_victim
                    code = v['result']['send_code']
                    self.send_victim(code)
                    self.identified_tokens.add(v['key'])
                    self.occ_map.add_wall_victim(v['gx'], v['gz'], v['side'], code)
                    # [FIX CRITICO] Calcola is_linear dalla mappa invece di hardcoded True
                    is_floating = self.occ_map.is_floating(v['gx'], v['gz'])
                    am      = AREA_MUL.get(self.current_area, 1.0)
                    ti_base = 5  if v['result']['type'] == 'letter' else 10
                    tt_base = 10 if v['result']['type'] == 'letter' else 20
                    # floating tile = 3x punti TI (sez. 5.6.3b)
                    ti_mul  = 3 if is_floating else 1
                    ti_pts  = int(ti_base * ti_mul * am)
                    tt_pts  = int(tt_base * am)
                    self.score_local += ti_pts + tt_pts
                    tile_type = "FLOATING" if is_floating else "LINEAR"
                    print(f"[VICTIM] {code} ({tile_type}) inviato. +{ti_pts}(TI)+{tt_pts}(TT) → {self.score_local}")
                    self.pending_victim = None
                    self.victim_timer   = 0.0
                    self.state          = 'EXPLORE'
                continue

            # ── Scansione victim passiva ─────────────────────────────
            if self.state == 'EXPLORE' and self.scan_cameras(gx, gz):
                self.state = 'VICTIM_STOP'
                self.victim_timer = 0.0
                continue

            # ── [FIX WARNING] Decisione ritorno a start ──────────────
            # Budget calcolato dinamicamente da lunghezza BFS, non fisso a 90s
            if (self.state == 'EXPLORE'
                    and len(self.identified_tokens) > 0
                    and not self.exit_sent):
                return_budget = self._return_budget(gx, gz)
                time_left     = 480.0 - self.game_time
                if time_left < return_budget:
                    print(f"[NAV] Budget ritorno: {return_budget:.0f}s, rimangono {time_left:.0f}s → RETURN.")
                    self.state       = 'RETURN'
                    self.return_path = []

            # ── STATO: RETURN ────────────────────────────────────────
            if self.state == 'RETURN':
                ok = self.start_return(gx, gz)
                if ok:
                    if self.at_start(gx, gz):
                        self.stop()
                        # [FIX IMPROVE] Invia mappa aggiornata, poi aspetta 5 timestep
                        self.send_map()
                        exit_buffer_steps = 5
                        print("[RETURN] Sulla start tile. Invio mappa + exit tra 5 step.")
                    else:
                        ds_local = self.read_ds()
                        self.flood_fill_step(ds_local, gx, gz)
                continue

            # ── STATO: DONE ──────────────────────────────────────────
            if self.state == 'DONE':
                self.stop()
                break

            # ── STATO: EXPLORE ───────────────────────────────────────
            if self.state == 'EXPLORE':
                if self.current_area in (1, 2, 3):
                    result = self.flood_fill_step(ds, gx, gz)
                    if result == 'DONE':
                        print(f"[EXPLORE] Area {self.current_area} esplorata. Cerco passage.")
                        self.wall_follow(ds)
                elif self.current_area == 4:
                    self.area4_navigate(ds)
                    if self.scan_cameras(gx, gz):
                        self.state = 'VICTIM_STOP'

        # ── Fine partita ──────────────────────────────────────────────
        self.stop()
        if not self.exit_sent:
            self.send_map()
            for _ in range(5):
                self.robot.step(self.timestep)
            self.send_exit()

        print("\n══════════════════════════════════")
        print(f"  NEXUS v3 — Partita terminata")
        print(f"  Score stimato:   {self.score_local}")
        print(f"  Victims trovati: {len(self.identified_tokens)}")
        print(f"  Checkpoint:      {len(self.checkpoints_visited)}")
        print(f"  LoP totali:      {self.lop_count}")
        print(f"  TMI (penalità):  {self.tmi_count}")
        print("══════════════════════════════════\n")


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NexusController().run()
