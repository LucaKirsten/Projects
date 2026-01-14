# Vision-Based Vehicle Tracking & Localization

This repository contains my final-year engineering project: a **vision-based tracking and localization system** for an **F1/10 autonomous vehicle**, using an **overhead camera** to localize the car within a **self-generated occupancy grid map**.

The project focuses on **practical visual localization** and **map construction** from real camera data, without relying on probabilistic filtering frameworks.

---

## Project Overview

The system processes an overhead video feed of a small-scale racing track and:

* Detects and tracks a single F1/10 vehicle
* Builds an **occupancy grid** representation of the track environment
* Localizes the vehicle within this grid over time

All localization is performed directly from visual measurements and geometric transformations.

---

## System Pipeline

```
Overhead Camera
      │
      ▼
Image / Video Frames
      │
      ▼
Vehicle Detection & Tracking
      │
      ▼
Ground-Plane Projection
      │
      ▼
Occupancy Grid Generation
      │
      ▼
Vehicle Localization (Grid Coordinates)
```

---

## Key Components

### Overhead Vision Input

* Fixed overhead camera view of the track
* Frame preprocessing and perspective handling
* Assumption of a planar track surface

---

### Vehicle Tracking

* Visual detection of the F1/10 vehicle
* Temporal tracking across frames
* Robust to short occlusions and noise

---

### Occupancy Grid Mapping

* Incremental construction of a 2D occupancy grid
* Grid represents free space and track boundaries
* Generated directly from visual observations

---

### Localization

* Vehicle position expressed in occupancy grid coordinates
* Uses geometric projection from image space to grid space
* No Kalman filtering or probabilistic state estimation

---

## Design Philosophy

* **Simplicity over complexity**: direct methods where possible
* **Vision-first**: all information derived from camera input
* **Deterministic pipeline**: no learned or probabilistic estimators
* **Real-data focused**: designed around physical track constraints

---

## Applications

* F1/10 autonomous racing research
* Vision-based localization
* Occupancy grid mapping from overhead cameras
* Offline analysis and replay

---

## Academic Context

This project was developed as a **final-year engineering project**, integrating:

* Computer vision
* Coordinate transformations
* Mapping and localization concepts
* Systems integration

---

## Author

**Luca Kirsten**

Engineering • Computer Vision • Autonomous Systems
