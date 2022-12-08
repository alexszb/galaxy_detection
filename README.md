# Galaxy detection

This is a project to detect and classify galaxies on sky images. It can show their main class type (spiral / elliptical) and also detect galactic rings.

The current 4 main modules are: sky image generator, Faster R-CNN neural network, DBSCAN - CNN, GUI

Sky image generator
- Downloads the target galaxy types from the SDSS telescope sky-server after filtering a galaxy catalogue.
- Applies a logarithmic brigthness correction on the galaxy images.
- Using average brigthness tiles crops out the corrected galaxies by their edges.
- Generates sky images by modifying randomly and putting these clean galaxy images on "galaxyless" sky images. Also saves annotations for later training.

Faster R-CNN
- Implementation follows mainly the original paper, but it uses RoI Align instead of RoI Pooling for better accuracy. The training is also done in two steps, first the RPN network then the Fast-RCNN network with the trained RPN weights.
- The most compute intensive functions are using GPU, such as box overlapping or RoI Align.

DBSCAN - CNN
- Galaxy detection is done by DBSCAN algorithm
- Classifications using two CNNs

GUI
- Allows you to open any sky image and with the trained neural network shows the detected galaxies in bounding-boxes. The colors represent the galaxy types: spiral without ring (blue), elliptical without ring (red), spiral with ring (green), elliptical with ring (cyan).
