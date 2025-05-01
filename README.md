# Cosmic Ray Muon Trigger System Simulation

This is Cosmic Ray Muon Trigger System Simulation, a small project I got the chance to work on with my colleage and friend Georgios Koryfidis. The project was done in the context of the course "Graduate Studies in Experimental Data Analysis" and its goal is to simulate cosmic ray muons that pass through two square surfaces acting like detectors.

We simulate muon events that are generated uniformly across the top surface of a detector. Each muon is assigned two angles:

θ (theta or zenith angle): This describes how steeply the muon travels. It is not generated uniformly; instead, it follows a cos²(θ) distribution, meaning muons are more likely to travel close to vertically.

φ (phi or azimuthal angle): This describes the direction around the vertical axis and is generated uniformly, so all directions are equally likely.

Once a muon is generated on the top surface, we check if its track intersects the bottom surface of the detector. This depends on its zenith angle:

If θ ≤ θ_max (a specific maximum angle), then the muon's trajectory reaches the bottom detector.

In this case, we consider it a trigger event because both the top and bottom detectors are hit.
