# Neuromorphic Projects

Projects on **event-based vision** and **spiking neural networks (SNNs)**.

## Contents

- [Background: event cameras and SNNs](#1-event-cameras): a short introduction to the ideas behind the projects
- [Hardware: OpenMV N6 + GENX320](#5-hardware-openmv-n6--genx320): getting events from the lab camera
- [SNN libraries and v2e](#6-snn-libraries-and-v2e): tools for training SNNs and simulating events
- [Project ideas](#7-project-ideas): seven proposed projects

## References

The background sections are based on two papers:

- **[1]** Ren et al., "A Comprehensive Survey on Event Camera Representation Learning", arXiv:2606.23078, 2026.
- **[2]** Yamazaki et al., "Spiking Neural Networks and Their Applications: A Review", *Brain Sciences* 12:863, 2022. [doi:10.3390/brainsci12070863](https://doi.org/10.3390/brainsci12070863)

LIF maths and code are from:

- **[3]** J. K. Eshraghian, [snnTorch Tutorial 2 – The Leaky Integrate-and-Fire Neuron](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html).

---

## 1. Event cameras

A normal camera captures full frames at a fixed rate. That causes motion blur, struggles with very bright or dark scenes, and produces lots of redundant data. An **event camera** works like the retina: each pixel acts on its own and reports only when its brightness changes [1].

Each pixel tracks the log of the light hitting it. When that value changes by more than a threshold *C* since the pixel's last event, the pixel fires an **event** [1]:

$$
|\Delta \log I| \geq C
$$

Each event is a tiny record:

$$
e = (x, y, t, p)
$$

where *x*, *y* is the pixel, *t* is a microsecond timestamp, and *p* is the **polarity**: +1 if the pixel got brighter (ON), −1 if it got darker (OFF).

This gives event cameras microsecond latency, high dynamic range, no motion blur and low power. A still scene produces almost no data [1].

---

## 2. Turning events into something a network can use

Events are sparse, arrive at irregular times, and come in variable numbers. That doesn't match the fixed-size images that normal neural networks expect. So the key design choice is the **representation**: how you package events before feeding them to a model [1].

There are two broad families [1].

**Dense representations** place events on a regular grid so standard CNNs can use them. They lose some timing detail.

| Representation | What each pixel stores |
| --- | --- |
| Event frame | Number of ON and OFF events in a time window |
| Time surface | How recently the pixel fired, fading over time: exp(−(*t* − *t*_last)/τ) |
| Binary map | Whether the pixel fired at all (1 or 0) |
| Voxel grid | Event counts split into several time bins (a 3D volume) |
| Learned | A small network learns how to bin the events |

**Sparse representations** keep events as events, treating them as tokens, graph nodes or a 3D point cloud. They preserve timing but need more specialised models.

The survey's main point is that no representation is best overall. Each trades timing detail against efficiency and compatibility with standard networks. Simple ones, such as event frames and binary maps, remain the practical choice when compute is limited [1].

---

## 3. Spiking neural networks

In the brain, neurons communicate with short electrical pulses called **spikes**. Each neuron collects input, and when its internal voltage (the **membrane potential**) crosses a threshold, it fires a spike and resets [2].

A normal artificial neuron outputs a single number, roughly an average firing rate. A **spiking neural network (SNN)** keeps the timing: each neuron holds a voltage that changes over time and outputs only spikes (1 or 0) [2].

| | Normal neural networks | Spiking neural networks |
| --- | --- | --- |
| Neurons output | Real numbers | Spikes (0 or 1) |
| Time | Ignored | Built in |
| Best hardware | GPUs, NPUs | Neuromorphic chips (Loihi, SpiNNaker, TrueNorth) |

SNNs are sparse and energy-efficient on the right hardware, and suit time-based data. They are harder to train [2].

### The LIF neuron

Several neuron models exist, ranging from very realistic but expensive (Hodgkin–Huxley) to very cheap. The most widely used is the cheap **leaky integrate-and-fire (LIF)** model [2]. It works in three steps:

1. **Integrate:** add up the incoming input.
2. **Leak:** let the voltage slowly drain back toward rest.
3. **Fire:** if the voltage crosses the threshold, output a spike and reset.

The maths and code below follow the snnTorch LIF tutorial [3].

**The membrane as an RC circuit.** A neuron's membrane behaves like a capacitor *C* (it stores charge) in parallel with a resistor *R* (ions leak through it). Injecting a current *I*_in(*t*) gives:

$$
\tau \frac{dU(t)}{dt} = -U(t) + R\, I_{\text{in}}(t), \qquad \tau = RC
$$

*U* is the membrane potential and τ is the time constant. With no input, the potential decays exponentially from its starting value *U*₀:

$$
U(t) = U_0\, e^{-t/\tau}
$$

With a constant input current, it rises toward *I*_in*R* instead. Starting from *U* = 0:

$$
U(t) = I_{\text{in}} R \left(1 - e^{-t/\tau}\right)
$$

**Discrete time.** To simulate this on a computer, step forward by a small Δ*t* each time (the forward Euler method):

$$
U(t + \Delta t) = U(t) + \frac{\Delta t}{\tau}\big(-U(t) + R\, I_{\text{in}}(t)\big)
$$

**Fire and reset.** When *U* exceeds a threshold *U*_thr, the neuron outputs a spike *S* = 1. Without a reset it would keep firing on every step. The default fix is **reset by subtraction**, which subtracts the threshold each time the neuron fires:

```python
def leaky_integrate_and_fire(mem, cur=0, threshold=1, time_step=1e-3, R=5.1, C=5e-3):
    tau_mem = R * C
    spk = (mem > threshold)                                  # spike if above threshold
    mem = mem + (time_step / tau_mem) * (-mem + cur * R) - spk * threshold  # leak, integrate, reset
    return mem, spk
```

Running it over time with a step of input current:

```python
import torch
num_steps = 200
cur_in = torch.cat((torch.zeros(10), torch.ones(190) * 0.2), 0)  # current switches on at t = 10
mem = torch.zeros(1)
spk_rec = []

for step in range(num_steps):
    mem, spk = leaky_integrate_and_fire(mem, cur_in[step])
    spk_rec.append(spk)
```

A stronger input current, or a lower threshold, makes the neuron fire more often.

There are two common reset options:

- **Subtract** (default): keeps any amount by which the potential overshot the threshold, so less information is lost.
- **Zero**: forces *U* back to 0, which gives sparser spikes.

snnTorch provides this whole model as a built-in neuron:

```python
import snntorch as snn
lif = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3)   # add reset_mechanism="zero" for a hard reset
for step in range(num_steps):
    spk, mem = lif(cur_in[step], mem)
```

### Turning data into spikes

Normal data has to be converted into spikes first [2]:

- **Rate coding:** a larger value fires more spikes.
- **Latency coding:** a larger value fires earlier.

Event cameras skip this step: their output is already spike-like [2].

### Training

A spike is an on/off step, so it has no usable gradient and normal backpropagation doesn't work directly. The main workarounds are [2]:

- **Surrogate gradients:** use the real spike going forward, but a smooth curve in its place when computing gradients.
- **STDP:** a brain-inspired rule. If an input spike comes just before an output spike, that connection gets stronger; if it comes just after, it gets weaker.
- **ANN-to-SNN conversion:** train a normal network, then convert it.

---

## 4. Why they fit together

An event is already a spike: a single binary signal at one pixel and one moment. So events can feed an SNN directly, keeping the camera's timing and sparsity instead of flattening them into frames.

The LIF neuron's leaky voltage also behaves much like a time surface: recent input counts more, and older input fades away.

---

## 5. Hardware: OpenMV N6 + GENX320

The OpenMV N6 with the GENX320 event camera is programmed using [OpenMV IDE](https://openmv.io/pages/download).

1. **Connect the camera** over USB and click **Connect** in the IDE. Make sure you use a data cable, not a charge-only cable.
2. **Check the firmware version** in the IDE's status bar at the bottom. If it is older than the IDE's version, the IDE will usually offer to update it automatically when you connect. Accept.
3. **Run a script:**
   - In OpenMV IDE, go to **File → Examples → Camera → HelloWorld → helloworld.py**. This script is written for the RGB camera and shows a live preview; it also works with the GENX320, where it shows a grayscale view.
   - To simply print the event streams, use [genx320_print_events.py](genx320_print_events.py) from this repository.

More details: [OpenMV N6 docs](https://docs.openmv.io/dev/openmvcam/quickref/openmv-n6.html), [OpenMV GENX320 docs](https://docs.openmv.io/dev/openmvcam/sensors/genx320.html), [OpenMV Projects](https://github.com/openmv/openmv-projects/tree/master)

---

## 6. SNN libraries and v2e

### SNN libraries

These Python libraries are built on PyTorch, so SNNs are trained much like normal networks: on a PC or GPU, with surrogate gradients.

| Library | What it is for |
| --- | --- |
| [snnTorch](https://snntorch.readthedocs.io/) | Beginner-friendly SNN library with step-by-step tutorials. The LIF code in Section 3 uses it [3]. |
| [SpikingJelly](https://github.com/fangwei123456/spikingjelly) | Full SNN framework with fast CUDA kernels and built-in loaders for neuromorphic datasets (e.g. DVS Gesture, N-MNIST) that convert events to frames. Many SNN papers release their code on it. |

A good starting point is snnTorch for learning, then SpikingJelly for larger models.

### v2e: turning video into events

Some projects use normal video datasets. [v2e](https://github.com/SensorsINI/v2e) converts ordinary video into **simulated event streams** that behave like a real event camera, including per-pixel threshold variation and noise. It can also interpolate frames to get finer timing than the original video.

---

## 7. Project ideas

| # | Project | Data | Event source |
| --- | --- | --- | --- |
| 1 | Evaluating CNN explanations with ReX | FEOT | Real |
| 2 | Day-to-night event-rate normalization | DailyDVS-200 | Real |
| 3 | Facial emotion recognition | NEFER | Real |
| 4 | Surveillance anomaly detection | UCF-Crime-DVS | Real |
| 5 | LED signal decoding | E-VLC | Real |
| 6 | Fish detection and tracking | MFT25 | Simulated (v2e) |
| 7 | Deepfake detection | FaceForensics++ | Simulated (v2e) |

### 1. Evaluating CNN Explanations Against Event-Camera Evidence Using ReX

Explainability methods aim to reveal which image regions influence a deep learning model's predictions, but visually plausible explanations do not necessarily reflect meaningful evidence used by the model. This project will investigate whether explanations generated by convolutional neural networks (CNNs) correspond to independently recorded visual activity captured by an event camera. Using the [Frame-Event Occlusion Tracking (FEOT) dataset](https://feot-dataset.github.io/), which provides spatially and temporally aligned RGB frames, event streams, object bounding boxes, and occlusion annotations, a CNN such as ResNet-50 will be trained to classify object categories from annotated RGB image crops. The [ReX explainability framework](https://github.com/ReX-XAI/ReX), a black-box tool for generating causal responsibility explanations for image classifiers, will identify the image regions responsible for each prediction. The resulting attribution maps will be compared with event-density maps computed from the synchronized event streams, which represent the spatial distribution of brightness changes over time. Spatial correlation, top-(k) overlap, and distributional similarity metrics will quantify the alignment between model explanations and event activity. The analysis will examine how explanation alignment and faithfulness vary across object categories and occlusion levels. The objective is to determine whether independently recorded event-camera activity provides useful complementary evidence for evaluating CNN explanations and whether the observed alignment corresponds to genuine model reliance rather than coincidental visual structure.

### 2. Automated Event-Rate Normalization for Day-to-Night Domain Generalization in Spiking Neural Networks

Event cameras offer high temporal resolution and high dynamic range, yet event-based neural networks can still experience substantial performance degradation under changing illumination. DailyDVS-200, a dataset comprising 22,046 event recordings across 200 action classes and 47 participants, provides an opportunity to investigate this problem using real event-camera data. Its published benchmark reports that Spikformer's action-recognition accuracy decreases from 38.30% on daytime samples to 19.81% on nighttime samples, although these results reflect subgroup performance rather than a controlled daytime-only training protocol ([Wang et al., 2024](https://arxiv.org/abs/2407.05106)). This project will investigate whether automated input normalization can reduce day-to-night domain shift in spiking neural networks (SNNs). First, daytime and nighttime event representations will be analyzed by comparing their means, variances, event sparsity, polarity distributions, and temporal activity. A compact SNN, using an established architecture such as Spikformer or the Spike-Driven Transformer (SDT), will then be trained on daytime recordings and evaluated on held-out daytime and nighttime samples using a controlled, recording-independent split. Per-window event-rate normalization will be applied as a lightweight, label-free adaptation strategy, rescaling each input according to its own event activity without requiring nighttime labels or updates to the model weights. The normalized model will be compared with an unnormalized baseline, as well as a conventional neural-network baseline. The objective is to determine whether differences in event-input statistics contribute to the observed performance gap and whether simple normalization can improve nighttime generalization.

### 3. Event-Based Facial Emotion Recognition with Spiking Neural Networks

Facial expressions change rapidly, and subtle facial movements can be difficult to capture using conventional frame-based cameras. Event cameras record asynchronous brightness changes at high temporal resolution, making them suitable for analyzing rapid facial motion. The [NEFER dataset](https://github.com/miccunifi/NEFER) (Berlincioni et al., 2023) provides paired RGB and real event-camera recordings of human faces, with emotion labels, face bounding boxes, and facial landmarks. Spiking neural networks (SNNs) have previously been applied to facial expression recognition using simulated event data, including the [Spiking-FER approach](https://arxiv.org/abs/2304.10211) (Barchid et al., 2023), but evaluating these methods on real event-camera recordings remains a relevant direction of investigation. This project will evaluate whether a compact SNN can recognize facial emotions directly from real event data. Event streams will be cropped to face regions using the provided bounding boxes and converted into time-binned event representations suitable for SNN processing. A compact SNN will be trained to classify facial emotions and compared with a conventional CNN trained on the same event representation and a CNN trained on paired RGB frames. A subject-independent split will ensure that individuals in the test set are excluded from training. The objective is to determine whether SNN-based processing of real event-camera recordings provides a practical approach to facial emotion recognition and how it compares with conventional frame-based alternatives.

### 4. Unsupervised Anomaly Detection in Event-Based Surveillance Video Using Spiking Autoencoders

Automated anomaly detection in surveillance video can help identify incidents such as fights, accidents, or theft, but anomalous events are rare, diverse, and costly to label. Event cameras record changes in brightness rather than conventional video frames, emphasizing motion while reducing redundant static-background information. [UCF-Crime-DVS](https://ojs.aaai.org/index.php/AAAI/article/view/32705) (Qian et al., 2025) introduces an event-based video anomaly detection benchmark derived from real-world surveillance footage. Existing event-based methods, including the spiking MSF network introduced with the dataset use supervised training with labeled normal and anomalous videos. This project will instead investigate a one-class approach in which a compact spiking convolutional autoencoder is trained exclusively on normal surveillance footage to reconstruct spatially downsampled event frames. Reconstruction error will serve as an anomaly score during evaluation, testing the hypothesis that unfamiliar motion patterns are more difficult to reconstruct. The SNN autoencoder will be compared with a conventional autoencoder trained on the same event representation. The objective is to determine whether SNN-based reconstruction can detect anomalies in real event-camera surveillance data without requiring labeled anomalous examples.

### 5. Spiking Neural Networks for Robust LED Signal Decoding from Event Cameras

Visible-light communication (VLC) uses modulated light sources, such as LEDs, to transmit information, offering potential applications in vehicle localization, robotics, and optical wireless communication. Event cameras provide an alternative to conventional cameras by recording asynchronous brightness changes, making them suitable for capturing rapid light transitions under varying illumination and camera motion. However, reliably decoding optical signals from event streams under different recording conditions remains challenging. This project will investigate whether spiking neural networks (SNNs) can decode LED identifiers from event-camera recordings and how their performance compares with conventional neural networks and signal-processing approaches. Using the publicly available [E-VLC dataset](https://woven-visionai.github.io/evlc-dataset/), which contains real event streams, LED annotations, and recordings under varying illumination, camera motion, and sensor sensitivity conditions, annotated event windows will be extracted and converted into suitable temporal representations for SNN processing. A compact SNN will be trained to identify LED transmitters from their event sequences and evaluated against a conventional neural-network baseline and an existing signal-processing decoder where applicable. To reduce the risk of models learning shortcuts from LED locations or recording backgrounds, evaluation will separate recordings and examine performance across different conditions. The objective is to determine whether SNN-based temporal processing provides a practical approach to LED identification and optical signal decoding from event-camera data.

### 6. Event-Based Fish Detection and Tracking with Spiking Neural Networks

Monitoring fish in underwater video supports marine ecology and aquaculture, but underwater imaging is affected by low light, turbidity, and rapid, irregular animal motion. Event cameras offer high temporal resolution and high dynamic range, making them promising for underwater observation. However, publicly available real event-camera datasets for fish tracking remain scarce, and existing underwater event datasets focus primarily on optical flow rather than animal detection. This project will use the [MFT25 multiple-fish tracking dataset](https://arxiv.org/abs/2507.06400) (Li et al., 2025), which contains 15 underwater video sequences with more than 400,000 annotated fish bounding boxes. The original videos will be converted into simulated event streams using [v2e](https://github.com/SensorsINI/v2e), a video-to-event simulator that models brightness changes over time. A compact spiking neural network (SNN) will be trained to detect fish from short event windows. The SNN detector will be compared with a conventional neural-network detector trained on the same event representation. Evaluation will use held-out video sequences to prevent leakage between training and testing. The objective is to investigate whether SNN-based processing of simulated event streams provides a practical approach to underwater fish detection and tracking, with potential applications in ecological monitoring and aquaculture.

### 7. Deepfake Detection Using Spiking Neural Networks

Deepfake detection remains challenging as facial manipulation techniques become increasingly realistic, particularly when models must generalize to manipulation methods absent from the training data. Recent research has explored spiking neural networks (SNNs) for detecting fully AI-generated videos by leveraging temporal inconsistencies between consecutive frames ([Jang et al., 2026](https://arxiv.org/abs/2605.05895)). However, detecting fully AI-generated videos differs from detecting deepfakes in which faces are manipulated within otherwise authentic video footage. This project will investigate whether SNNs can improve the detection of manipulated facial videos, including face swapping and facial reenactment, compared with conventional neural networks. Using the publicly available [FaceForensics++ dataset](https://github.com/ondyari/faceforensics), which contains authentic videos and four types of facial manipulation, the original video sequences will be converted into simulated event streams using [v2e](https://github.com/SensorsINI/v2e), a video-to-event simulator that models brightness changes over time. The resulting event streams will serve as inputs to an SNN trained to distinguish authentic from manipulated videos. The objective is to determine whether SNN-based processing of event streams derived from video provides practical advantages for facial deepfake detection.
