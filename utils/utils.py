import ggplot as gp
import numpy as np


def draw_line_chart(data, x_name, y_name, title):
  p = gp.ggplot(gp.aes(x=x_name, y=y_name), data=data) + gp.geom_line(color='blue') + gp.ggtitle(title)
  print(p)

def save_results(data, file_path):
  np.savetxt(file_path, data, fmt='%d')