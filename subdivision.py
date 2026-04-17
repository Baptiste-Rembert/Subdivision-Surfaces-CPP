import numpy as np
import trimesh 
from tqdm import tqdm
from ipywidgets import widgets, IntSlider
from IPython.display import display
import plotly.graph_objects as go
from utils import *
import meshio
import heapq
from collections import deque

def control_points_Chaikin_2D(Points, closed=True):
    n = len(Points)
    new_points = []
    for i in range(n if closed else n-1):
        p0 = Points[i]
        p1 = Points[(i + 1) % n]  # Wrap around to the first point
        q = 0.75 * p0 + 0.25 * p1  # First new point
        r = 0.25 * p0 + 0.75 * p1  # Second new point
        new_points.append(q)
        new_points.append(r)
    return np.array(new_points)

def chaikin_subdivision(Points, iterations=1, closed=True):
    for _ in range(iterations):
        Points = control_points_Chaikin_2D(Points, closed)
    return Points

def control_points_DooSabin_3D(Vertices, Faces):
    new_vertices = []
    for face in Faces:
        face_vertices = Vertices[face]
        centroid = np.mean(face_vertices, axis=0)
        for vertex in face_vertices:
            new_vertex = (vertex + centroid) / 2
            new_vertices.append(new_vertex)
    return np.array(new_vertices)
