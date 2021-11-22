from manim import *
import math
import pandas as pd
import random


def add(v1, v2):
    return [v1[0] + v2[0], v1[1] + v2[1]]


def mul(v, s):
    return [v[0] * s, v[1] * s]


def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def squared_distance(a, b):
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])


class AABB:
    def __init__(self, center, width, height):
        self.center = center
        self.width = width
        self.height = height

    def contains(self, mob):
        return self.center[0] - self.width <= mob.pos[0] < self.center[0] + self.width and self.center[1] \
               - self.height <= mob.pos[1] < self.center[1] + self.height

    def intersects(self, rect):
        return not (self.center[0] - self.width >= rect.center[0] + rect.width or self.center[0] + self.width <=
                    rect.center[0] - rect.width or self.center[1] + self.height <= rect.center[1] - rect.height or
                    self.center[1] - self.height >= rect.center[1] + rect.height)

    def encompass(self, rect):
        return self.center[0] - self.width <= rect.center[0] - rect.width and self.center[1] - self.height <= \
               rect.center[1] - rect.height and rect.center[0] + rect.width <= self.center[0] + self.width and \
               rect.center[1] + rect.height <= self.center[1] + self.height


class QuadTree:
    def __init__(self, boundary, cap):
        self.boundary = boundary
        self.cap = cap
        self.array = []
        self.divided = False

    def subdivide(self):
        x = self.boundary.center[0]
        y = self.boundary.center[1]
        w = self.boundary.width / 2
        h = self.boundary.height / 2
        self.northwest = QuadTree(AABB((x - w, y + h, 0), w, h), self.cap)
        self.northeast = QuadTree(AABB((x + w, y + h, 0), w, h), self.cap)
        self.southeast = QuadTree(AABB((x + w, y - h, 0), w, h), self.cap)
        self.southwest = QuadTree(AABB((x - w, y - h, 0), w, h), self.cap)
        self.divided = True

    def insert(self, mob):
        if not self.boundary.contains(mob):
            return
        if len(self.array) < self.cap:
            self.array.append(mob)
        else:
            if not self.divided:
                self.subdivide()
            self.northwest.insert(mob)
            self.northeast.insert(mob)
            self.southeast.insert(mob)
            self.southwest.insert(mob)

    def query(self, rect):
        found = []
        if not self.boundary.intersects(rect):
            return found
        if rect.encompass(self.boundary):
            found.extend(self.array)
        else:
            for p in self.array:
                if rect.contains(p):
                    found.append(p)
        if self.divided:
            found.extend(self.northwest.query(rect))
            found.extend(self.northeast.query(rect))
            found.extend(self.southeast.query(rect))
            found.extend(self.southwest.query(rect))
        return found


class Particle(Dot):
    def __init__(self, position_init, designation=None, vector_init=[0, 0], **kwargs):
        self.pos = position_init
        self.vector = vector_init
        self.designation = designation
        Dot.__init__(self, point=np.array(position_init + [0]), **kwargs)


class Wall:
    def __init__(self, start_point, end_point):
        self.start = [start_point[0], start_point[1]]
        self.end = [end_point[0], end_point[1]]
        self.pos = [(start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2]
        self.length = math.sqrt(squared_distance(start_point, end_point))
        minimum = 0
        if end_point[0] - start_point[0] == 0:
            minimum += 0.000001
        self.slope = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0] + minimum)


class scene_2(MovingCameraScene):
    counter = 0

    def construct(self):
        self.camera.frame.shift(DOWN * 0.5)
        df1 = pd.read_csv("data/ScatterArt.csv", header=None)
        df1 = df1.values.tolist()
        dots = VGroup()
        active = VGroup()
        elements = []
        i = 0
        while i < len(df1):
            df1[i] = [x for x in df1[i] if str(x) != 'nan']
            color = df1[i][1]
            for j in range(2, len(df1[i])):
                if squared_distance([float(df1[i][j]), float(df1[i + 1][j])], [6, -3]) < 16:
                    elements.append(self.counter)
                    active.add(Particle([float(df1[i][j]), float(df1[i + 1][j])], fill_color=color, radius=0.025))
                    self.counter += 1
                else:
                    dots.add(Particle([float(df1[i][j]), float(df1[i + 1][j])], fill_color=color, radius=0.025))
            i += 2
        self.add(dots)
        print(self.counter)
        text = Text("Data told with a story", font="Futura", font_size=72).shift(DOWN * 3.5)
        self.add(text)
        radius = ValueTracker(1)

        def area(point):
            x = point[0]
            y = point[1]
            return (x - 6) * (x - 6) + (y + 3) * (y + 3) < radius.get_value() * radius.get_value()

        def gravity(mob, dt):
            to_remove = []
            for i in range(self.counter):
                x_cor = (30 * (7 - mob[i].pos[0]) / squared_distance(mob[i].pos, [7, -5]) + 1) * area(mob[i].pos)
                y_cor = (25 * (-5 - mob[i].pos[1]) / squared_distance(mob[i].pos, [7, -5])) * area(mob[i].pos)
                mob[i].vector = add(mob[i].vector, mul([x_cor, y_cor], dt))
                mob[i].pos = add(mob[i].pos, mul(mob[i].vector, dt))
                mob[i].move_to(np.array(mob[i].pos + [0]))
                if mob[i].pos[1] < -5:
                    to_remove.append(mob[i])
            mob.remove(*to_remove)
            self.counter -= len(to_remove)
            if radius.get_value() < 4:
                radius.increment_value(dt)
        active.add_updater(gravity)
        self.add(active)
        self.play(self.camera.frame.animate.move_to((6, -2.5, 0)).set_width(1), run_time=3)
        self.wait(2)


class scene_3(MovingCameraScene):
    g = -9.8
    p_radius = 0.05
    counter = 0
    rough_prop = 0.1

    def construct(self):
        curve = SVGMobject("assets/svg_images/test svg", stroke_opacity=1, stroke_color=WHITE, stroke_width=0.02,
                           fill_opacity=0).scale(10.5)
        #quadrect = AABB(center=[0, 1], width=19.5, height=22.5)
        #curve.append_points(curve.get_all_points())
        #borders = []
        #side_length = 2 * self.p_radius + 0.001
        #for i in [8, 9, 3, 4, 2, 7, 6]:
        #    if i in [8, 9]:
        #        curve[i].set_stroke(width=2)
        #    else:
        #        curve[i].set_stroke(width=1)
        #    lines = []
        #    start_point = 0
        #    for j in np.linspace(start=0, stop=1, num=250):
        #        if j == 0:
        #            continue
        #        wall = Wall(curve[i].point_from_proportion(start_point), curve[i].point_from_proportion(j))
        #        side_length = max(side_length, wall.length)
        #        lines.append(wall)
        #        start_point = j
        #    borders.append(lines)
        #df1 = pd.read_csv("data/ScatterArt.csv", header=None)
        #df1 = df1.values.tolist()
        #particles = VGroup()
        #i = 0
        #while i < len(df1):
        #    df1[i] = [x for x in df1[i] if str(x) != 'nan']
        #    for j in range(2, int(self.rough_prop * len(df1[i]))):
        #        p = Particle([-6 + 12 * random.random(), 7.5 + 2 * random.random()], designation=df1[i][0],
        #                               radius=self.p_radius, fill_color=df1[i][1])
        #        particles.add(p)
        #        self.counter += 1
        #    i += 2

        def match(particle, wall):
            if wall in borders[0] + borders[1]:
                return True
            elif particle.designation == 'Sky' and wall in borders[6]:
                return True
            elif particle.designation == 'Mountain' and wall in borders[5]:
                return True
            elif particle.designation == 'Clouds' and wall in borders[4]:
                return True
            elif particle.designation == 'Grass' and wall in borders[3]:
                return True
            elif particle.designation == 'Trees' and wall in borders[2]:
                return True
            else:
                return False

        def updater(mob, dt):
            steps_per_frame = [5, 30]
            AABBs = [AABB(center=[0, 6.125], width=9.75, height=6.125),
                     AABB(center=[0, -6.125], width=9.75, height=6.125)]
            for v in range(2):
                for j in range(steps_per_frame[v]):
                    ptree = QuadTree(AABBs[v], 2)
                    bounded_mobjects = []
                    for i in range(self.counter):
                        if AABBs[v].contains(mob[i]):
                            ptree.insert(mob[i])
                            bounded_mobjects.append(i)
                    for b in borders:
                        for l in b:
                            ptree.insert(l)
                    for i in bounded_mobjects:
                        neighbors = ptree.query(AABB(center=mob[i].pos, width=side_length, height=side_length))
                        for m in neighbors:
                            if m in particles:
                                if mob[i].pos == m.pos:
                                    continue
                                delta = squared_distance(mob[i].pos, m.pos)
                                if delta < 4 * self.p_radius * self.p_radius:
                                    delta = math.sqrt(delta)
                                    unit_collision = mul(add(m.pos, mul(mob[i].pos, -1)), 1 / delta)
                                    error = self.p_radius - delta / 2 + 0.001
                                    v_2 = dot(unit_collision, mob[i].vector)
                                    v_1 = dot(unit_collision, m.vector)
                                    mob[i].pos = add(mob[i].pos, mul(unit_collision, -error))
                                    m.pos = add(m.pos, mul(unit_collision, error))
                                    mob[i].vector = add(mob[i].vector, mul(unit_collision, v_1 - v_2))
                                    m.vector = add(m.vector, mul(unit_collision, v_2 - v_1))
                            else:
                                if match(mob[i], m):
                                    delta = (-m.slope * mob[i].pos[0] + mob[i].pos[1] + m.slope * m.start[0] -
                                             m.start[1])
                                    delta = delta * delta / (m.slope * m.slope + 1)
                                    if delta < self.p_radius * self.p_radius:
                                        delta = math.sqrt(delta)
                                        unit_collision = mul([m.end[1] - m.start[1], m.start[0] - m.end[0]], 1 /
                                                             m.length)
                                        if m in borders[1]:
                                            unit_collision = mul(unit_collision, -1)
                                        error = self.p_radius - delta + 0.001
                                        mob[i].pos = add(mob[i].pos, mul(unit_collision, error))
                                        vel = dot(unit_collision, mob[i].vector)
                                        mob[i].vector = add(mob[i].vector, mul(unit_collision, -1.5 * vel))
                        mob[i].vector = add(mob[i].vector, [0, self.g * dt / steps_per_frame[v]])
                        mob[i].pos = add(mob[i].pos, mul(mob[i].vector, dt / steps_per_frame[v]))
                for i in range(self.counter):
                    mob[i].move_to(np.array(mob[i].pos + [0]))
        self.add_foreground_mobject(curve)
        text1 = Text("Data", font_size=144, font="Futura").move_to((-0.25, 9.2, 0))
        text2 = Text("Data\nsorted", font_size=100, font="Futura").move_to((5.7, 1, 0))
        text3 = Text("Data\nvisualized", font_size=72, font="Futura").move_to((6.5, -6, 0))
        graph = VGroup(
            Rectangle(width=7, height=1.5, fill_color="#202124", fill_opacity=1, stroke_width=0).
                move_to((-0.2, -10, 0)),
            Line((-3.7, -9.25, 0), (3.3, -9.25, 0), stroke_width=2),
            Line((-3.7, -9.25, 0), (-3.7, -2.6, 0), stroke_width=2),
            *[Line((-3.8, i, 0), (-3.6, i, 0), stroke_width=1) for i in np.linspace(start=-9.25, stop=-2.6, num=6)
              if i != -9.25]
        )
        text4 = Text("Frequency", font_size=45, font="Futura").rotate(PI / 2).next_to(graph[2], LEFT, buff=0.1)
        self.camera.frame.set_width(15).shift(UP * 8 + LEFT * 0.25)
        self.add(text1, text2, text3)
        #particles.add_updater(updater)
        #self.add(particles)
        self.wait(3)
        self.play(self.camera.frame.animate.move_to((2, 1, 0)), run_time=2)
        self.wait(3)
        self.play(self.camera.frame.animate.move_to((2, -6, 0)), run_time=2)
        self.wait(1)
        self.play(AnimationGroup(FadeOut(curve), FadeIn(graph[0]),
                                 AnimationGroup(*[Create(graph[i]) for i in range(1, 8)], lag_ratio=0.2),
                                 Write(text4), lag_ratio=0.5), run_time=2)
        self.wait(2)
        self.add_foreground_mobject(graph[0])
        #particles.clear_updaters()

        def gravity(mob, dt):
            for i in range(self.counter):
                x_cor = 30 * (-5 - mob[i].pos[0]) / squared_distance(mob[i].pos, [-5, -11]) - 1
                y_cor = 15 * (-11 - mob[i].pos[1]) / squared_distance(mob[i].pos, [-5, -11])
                mob[i].vector = add(mob[i].vector, mul([x_cor, y_cor], dt))
                mob[i].pos = add(mob[i].pos, mul(mob[i].vector, dt))
                mob[i].move_to(np.array(mob[i].pos + [0]))
        #particles.add_updater(gravity)
        #self.add(particles)
        self.play(AnimationGroup(*[FadeOut(g) for g in graph], lag_ratio=0.2), FadeOut(text4),
                  self.camera.frame.animate.move_to((-7, -9.25, 0)).set_width(2), run_time=3, rate_func=smooth)


class sanggu_logo(Scene):
    def construct(self):
        curve = SVGMobject("assets/svg_images/sanggu_logo", stroke_opacity=0, fill_opacity=1,
                           fill_color=WHITE).scale(2.5)
        self.add(curve)