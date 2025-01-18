# AMIRwGA
this is a repository documenting my project 'Automatic Medical Image Registration with Genetic Algorithm' I worked on during my semester abroad in Taiwan at NTHU

#Image Registration and Genetic Algorithm
Medical image processing is a versatile and growing field of research that involves the use and
analysis of images depicting structures of the human body.[6] These images are typically ac-
quired using X-ray machines, computed tomography scanners (CT), magnetic resonance imag-
ing devices (MRI), or positron emission tomography scanners (PET). These imaging techniques
generate three-dimensional images by producing different types of waves, which capture the ex-
amined object layer by layer in various ways. X-ray and CT machines use X-ray radiation,
while MRI devices generate radio waves with the help of magnetic fields.[7] PET scanners vi-
sualize radioactive substances by detecting two gamma-ray photons simultaneously, which are
produced during the decay of a radionuclide.[8] Unlike X-ray, CT, and MRI images — which
depict bones, tissues, and organs themselves[7] — PET images visualize the metabolic activity
of tissues or structures in general.[8] In medical examinations, it is common for PET images to
be taken alongside MRI and CT scans. These procedures are essential for medical intervention
planning and disease diagnosis.[6] A key aspect of image processing is image registration. This
involves aligning images of different modalities — such as PET and CT images — or images of
the same modality taken at different times into a common coordinate system to overlay them.[9]
The goal is, on one hand, to highlight relevant structures (e.g. malignant tumors) through the
interplay of different modalities to enable accurate diagnoses; on the other hand, it is to repre-
sent the progression of diseases over time (e.g. cancer growth) to monitor the effectiveness of
therapies. In image registration, image pairs are considered. One of the images represents the
reference image (fR : # ! R), while the other is the template image (fT : #³
! R). The goal of
the registration is to find the optimal transformation (T : # ! #³) that makes the transformed
template image as similar as possible to the reference image.[9] This is why image registration is
subject to an optimization problem. Genetic algorithms are particularly well-suited for solving
optimization problems. They are evolutionary algorithms — based on natural evolutionary
principles (e.g. fitness, selection, mutation, recombination, etc.) — that simulate an artificial
evolution.[10]

#Projectgoal
The goal of the project was to design and implement a GA that identifies the optimal affine
transformation parameters for a 2D-2D image registration of CT and PET scans, aiming to
optimally align the images of a dataset — specifically of breast cancer patients — and visualize
the most metabolically active body regions (e.g., tumors) within their anatomical context (i.e.,
the breast). The runtime should be minimized, and the algorithm’s accuracy maximized in order
to achieve the highest possible convergence in the shortest possible time. To accomplish this,
the basic framework of the algorithm should be as flexible and modular as possible, allowing
for the variation of as many input parameters as needed to identify the best combinations and
thus achieve the highest accuracy.
