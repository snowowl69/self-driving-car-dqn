import math
import numpy as np

try:
    import pygame
except Exception:
    pygame = None

class DrivingEnv:
    """
    2D Self-Driving Car Lane Environment
    
    A custom reinforcement learning environment for training a self-driving car
    agent to navigate a lane. Supports both straight and curved (sine-wave) roads
    with adjustable difficulty levels.
    
    State Space (10 dimensions):
        - Normalized x position
        - Normalized y position
        - Normalized velocity
        - Normalized angle
        - Normalized distance from lane center
        - 5 normalized distance sensor readings (at -60°, -30°, 0°, 30°, 60°)
    
    Action Space (4 discrete actions):
        0 - Accelerate
        1 - Brake
        2 - Steer left
        3 - Steer right
    
    Reward Function:
        - Positive reward: 1.0 - 0.01 × |distance_from_center|
        - Termination penalty: -50.0 (when car leaves lane or goes out of bounds)
    
    Args:
        render: Whether to render the environment using pygame (default: False)
        fps: Frames per second for rendering (default: 60)
        curved: Use curved (sine-wave) road instead of straight (default: True)
        narrow: Use narrower lane width for increased difficulty (default: False)
    """
    def __init__(self, render=False, fps=60, curved=True, narrow=False):
        self.width = 800
        self.height = 400
        self.base_center = self.height / 2.0
        self.curved = curved         # NEW: straight vs curved
        self.lane_half = 40.0 if narrow else 80.0  # narrower lane if enabled
        self.max_speed = 8.0
        self.min_speed = 0.0
        self.dt = 1.0
        self.sensor_angles = [-60, -30, 0, 30, 60]
        self.max_sensor = 200.0
        self.render_mode = render
        self.fps = fps

        # render state
        self._pg_inited = False
        self.screen = None
        self.clock = None
        self.font = None

        self._sensor_vals_px = [self.max_sensor for _ in self.sensor_angles]

        if self.render_mode:
            self._init_pygame()

        self.reset()

    # ---------------- Lane definition ----------------
    def lane_center_at(self, x):
        if self.curved:
            # sine wave road: amplitude=60px, wavelength ~ 628 px
            return self.base_center + 60.0 * math.sin(0.01 * x)
        else:
            # straight lane
            return self.base_center

    # ---------------- Gym-like API ----------------
    @property
    def action_space(self):
        return 4

    @property
    def state_size(self):
        return 5 + len(self.sensor_angles)

    def reset(self):
        self.x = 50.0
        self.y = self.lane_center_at(self.x)
        self.vel = 2.0
        self.angle = 0.0
        self.steps = 0
        obs = self._get_obs()
        if self.render_mode:
            self._render()
        return obs

    def step(self, action):
        if action == 0:
            self.vel = min(self.max_speed, self.vel + 0.8)
        elif action == 1:
            self.vel = max(self.min_speed, self.vel - 1.2)
        elif action == 2:
            self.angle -= math.radians(6)
        elif action == 3:
            self.angle += math.radians(6)

        dx = math.cos(self.angle) * self.vel * self.dt
        dy = math.sin(self.angle) * self.vel * self.dt
        self.x += dx
        self.y += dy
        self.steps += 1

        center_y = self.lane_center_at(self.x)
        dist_center = self.y - center_y
        done = (
            abs(dist_center) > self.lane_half or
            self.x > self.width - 10 or
            self.x < 0
        )

        if done:
            reward = -50.0  # softer penalty for harder tasks
        else:
            reward = 1.0 - 0.01 * abs(dist_center)

        obs = self._get_obs()
        if self.render_mode:
            self._render()
        return obs, reward, done, {}

    # ---------------- Observation + sensors ----------------
    def _get_obs(self):
        center_y = self.lane_center_at(self.x)
        x_n = self.x / self.width
        y_n = self.y / self.height
        vel_n = self.vel / self.max_speed
        angle_n = (self.angle % (2 * math.pi)) / (2 * math.pi)
        dist_center = (self.y - center_y) / self.lane_half

        sensor_vals_px = []
        for a_deg in self.sensor_angles:
            ang = self.angle + math.radians(a_deg)
            dx = math.cos(ang)
            dy = math.sin(ang)
            y_top = center_y - self.lane_half
            y_bottom = center_y + self.lane_half
            t_candidates = []
            if abs(dy) > 1e-6:
                t_top = (y_top - self.y) / dy
                t_bottom = (y_bottom - self.y) / dy
                if t_top > 0:
                    t_candidates.append(t_top)
                if t_bottom > 0:
                    t_candidates.append(t_bottom)
            if not t_candidates:
                dist = self.max_sensor
            else:
                dist = min(t_candidates)
                dist = min(dist, self.max_sensor)
            sensor_vals_px.append(dist)

        self._sensor_vals_px = sensor_vals_px
        sensor_norm = [d / self.max_sensor for d in sensor_vals_px]
        obs = np.array([x_n, y_n, vel_n, angle_n, dist_center] + sensor_norm, dtype=np.float32)
        return obs

    # ---------------- Rendering ----------------
    def _init_pygame(self):
        if pygame is None:
            raise RuntimeError("pygame not installed. Run: pip install pygame")
        if not self._pg_inited:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("2D Self-Driving RL")
            self.clock = pygame.time.Clock()
            try:
                self.font = pygame.font.SysFont("consolas", 16)
            except Exception:
                self.font = None
            self._pg_inited = True

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                raise SystemExit

    def _render(self):
        self._handle_events()

        BG = (30, 30, 30)
        ROAD = (60, 60, 60)
        CAR = (80, 200, 120)
        SENSOR = (100, 180, 255)

        self.screen.fill(BG)

        # draw road as curved polyline if curved=True, else straight rectangle
        if self.curved:
            pts_top = []
            pts_bottom = []
            for x in range(0, self.width, 10):
                cy = self.lane_center_at(x)
                pts_top.append((x, int(cy - self.lane_half)))
                pts_bottom.append((x, int(cy + self.lane_half)))
            pygame.draw.polygon(self.screen, ROAD, pts_top + pts_bottom[::-1])
        else:
            y_top = int(self.base_center - self.lane_half)
            y_bottom = int(self.base_center + self.lane_half)
            pygame.draw.rect(self.screen, ROAD, pygame.Rect(0, y_top, self.width, y_bottom - y_top))

        # draw sensors
        for a_deg, dist in zip(self.sensor_angles, self._sensor_vals_px):
            ang = self.angle + math.radians(a_deg)
            ex = self.x + math.cos(ang) * dist
            ey = self.y + math.sin(ang) * dist
            pygame.draw.line(self.screen, SENSOR, (self.x, self.y), (ex, ey), 1)
            pygame.draw.circle(self.screen, SENSOR, (int(ex), int(ey)), 3)

        # draw car (triangle)
        L, W = 22.0, 12.0
        hx, hy = math.cos(self.angle), math.sin(self.angle)
        rx, ry = -math.sin(self.angle), math.cos(self.angle)
        front = (self.x + hx * L / 2, self.y + hy * L / 2)
        rear_left = (self.x - hx * L / 2 + rx * W / 2, self.y - hy * L / 2 + ry * W / 2)
        rear_right = (self.x - hx * L / 2 - rx * W / 2, self.y - hy * L / 2 - ry * W / 2)
        pygame.draw.polygon(self.screen, CAR, [(int(front[0]), int(front[1])),
                                               (int(rear_left[0]), int(rear_left[1])),
                                               (int(rear_right[0]), int(rear_right[1]))])

        if self.font:
            txt = f"x: {int(self.x)} vel: {self.vel:.2f} angle: {math.degrees(self.angle)%360:5.1f}°"
            surf = self.font.render(txt, True, (230, 230, 230))
            self.screen.blit(surf, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self._pg_inited:
            pygame.quit()
            self._pg_inited = False
