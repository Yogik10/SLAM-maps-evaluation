# SLAM-maps-evaluation
Evaluation of 2D laser SLAM maps by corners count, count of enclosed areas and a ratio of occupied cells to free

# How to use
For use in Linux OS you need to **unpack archive lib.zip** so that the "lib" folder is in the same directory as the program<br>
For run write **`python3 maps_evaluation.py <path/map_name.pgm>`**, where **<path/map_name.pgm>** is the path and the map name<br>
If you want to save processed images, add **"save"** parameter in the end of command like: **`python3 maps_evaluation.py map_fastslam.pgm save`**
**There are work examples in /pictures.**

# Demonstration
## Corners detection:
![Corners detection](https://github.com/Yogik10/SLAM-maps-evaluation/blob/main/pictures/map_fastslam_corners_part.png)

## Searched enclosed areas:
![Searched enclosed areas](https://github.com/Yogik10/SLAM-maps-evaluation/blob/main/pictures/map_hector_enclosed_areas_part.png)
