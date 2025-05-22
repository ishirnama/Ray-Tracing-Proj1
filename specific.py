import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

def magnitude(vector):
    return np.linalg.norm(vector)

def no_homo(vector):
    return vector[:3]

def scale_matrix(Sx,Sy,Sz,w):
    mat = np.array([
        [Sx,0,0,0],
        [0,Sy,0,0],
        [0,0,Sz,0],
        [0,0,0,w]
    ])
    return mat

def vector(x,y,z,w):
    v = np.array([[x],[y],[z],[w]])
    return v

def spawn_sphere(radius):
  global ax
  # Make data
  r = radius
  u = np.linspace(0, 2 * np.pi, 100)
  v = np.linspace(0, np.pi, 100)
  x = r * np.outer(np.cos(u), np.sin(v))
  y = r * np.outer(np.sin(u), np.sin(v))
  z = r * np.outer(np.ones(np.size(u)), np.cos(v))

  # Plot the surface
  ax.plot_surface(x, y, z, cmap='BuPu')

  # Set an equal aspect ratio
  ax.set_aspect('equal')
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  
def draw_sphere_on_hit(Q, d, radius, step_length, domain):
    start = domain[0]
    end = domain[1]
    # Initialize the 2D grid
    count = 0
    grid_size = int((radius*2) / step_length) + 1  # Considering the range from -4 to 4
    canvas = [[" " for _ in range(grid_size)] for _ in range(grid_size)]
    img = Image.new( 'RGB', (255, 255), "black") # Create a new black image
    pixels = img.load()
    # Iterate over the x and y values
    for i, z in enumerate(np.arange(start, end + step_length, step_length)):
        for j, x in enumerate(np.arange(start, end + step_length, step_length)):
            # Set the ray origin Q (x, y, z) with varying x and y
            Q = np.array([[x], [start], [z], [1]])

            # Coefficients of the quadratic equation at^2 + bt + c = 0
            a = magnitude(no_homo(d))**2
            b = 2 * np.dot(no_homo(d).T, no_homo(Q))[0][0]
            c = magnitude(no_homo(Q))**2 - radius**2

            # Discriminant
            discriminant = b**2 - 4*a*c

            # Check if the ray intersects the sphere
            if discriminant >= 0:
                pixels[i,j] = (255, 255, 255)
                count = count + 1
            else:
                pixels[i,j] = (0, 0, 0)
    # Print the 2D array
    print(f"diameter of the image = {grid_size}")
    print(f"total white pixels = {count}")
    return img
fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")

h = float(input("Enter separation distance  : "))
r = float(input("Enter radius length     : "))

w=1

domain = [-r,r]
t=scale_matrix(6,6,6  ,w)
# P(t) = Q + td0
d = vector(0,1,0,w)
Q = vector(-r,-r,-r,w)
spawn_sphere(r)
# spawn_frustum(2.0, 4.64, 9.92, 0, 0, 0, 1.230387597)
# spawn_vector_field(h, d, t, -r, -r, -r, domain)
# spawn_ring(2.0, 1.230387597)
img = draw_sphere_on_hit(Q, d, r, h, domain)
print(img)
img.show()
ax.view_init(30, 30)
plt.show()