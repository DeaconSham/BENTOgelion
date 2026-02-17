# BENTOgelion

A survival rover built for MakeUofT 2026. The robot scouts areas for resources (food, water, people) and maps them on a dashboard for remote monitoring.

- **Demo**: [YouTube Video](https://youtu.be/4KVLskjAxu8)
- **Devpost**: [Project Page](https://devpost.com/software/bentogelion)

## System Overview

The system is split into three parts:

1.  **The Rover (Raspberry Pi 4)**
    - Controls motors and streams video
    - Communicates with the backend over Wi-Fi

2.  **Vision Prosessing (Laptop)**
    - Runs YOLOv8 for object detection
    - Offloaded from the Pi due to performance constraints

3.  **Dashboard (Flask + JS)**
    - Displays a live map of the rover's path
    - Marks discovered resources in real-time

## Setup

### 1. Vision & Backend (Laptop)
Run the vision script to start detecting objects:
```bash
python vision.py
```

Start the web server:
```bash
cd backend
python backend.py
```
Open `http://localhost:5002` to view the dashboard.

### 2. Rover Control (Raspberry Pi)
SSH into the Pi and run:
```bash
cd pi
python bot.py
```

This script connects to the backend and begins the patrol.

## Tech Stack

- **Hardware**: Raspberry Pi 4, custom chassis
- **Vision**: YOLOv26
- **Backend**: Python (Flask)
- **Frontend**: HTML, JavaScript
