# BE 640 Medical Image Analysis (Fall 2026)

Welcome to the **BE 640: Medical Image Analysis** course.

This course offers both theoretical and practical knowledge about the theory of stochastic and geometric models of medical imaging. We will explore spatial interaction models, intensity models, and geometric shape models, and how they are applied to solve complex biomedical problems. The transition towards computational medical imaging allows bioengineers and AI models to assist radiologists and healthcare professionals in managing, diagnosing, and treating various diseases.

We will explore how advanced mathematical modeling and image processing are revolutionizing bioengineering, particularly in analyzing and interpreting modalities such as CT, MRI, Ultrasound, PET, and SPECT. Join us as we uncover the latest advancements and methodologies in this exciting intersection of technology, mathematics, and healthcare. Whether you are a student, researcher, or simply curious about the future of healthcare technology, this repository offers valuable insights into the innovative applications of stochastic and geometric models in medical image analysis.

> If you encountered any issues or errors in the code or lectures, please feel free to let me know. I will be more than happy to fix them and update the repository accordingly. Your feedback is highly appreciated and will help me improve the quality of the content provided in this series.

## Programming Language and Libraries

This course heavily utilizes **MATLAB** for medical image processing, stochastic modeling, and geometric analysis. All scripts and functions require MATLAB (R2022b or newer recommended) with the **Image Processing Toolbox** installed.

You can verify your MATLAB installation and toolboxes by running the following command in the MATLAB Command Window:

```matlab
ver
```

Ensure that `Image Processing Toolbox` and `Statistics and Machine Learning Toolbox` are listed in the output.

## Environment Setup

Unlike Python-based courses, this repository relies on MATLAB. To set up your workspace:

1. Clone or download this repository to your local machine.
2. Open MATLAB and navigate to the root folder of the repository.
3. Add the repository and its subfolders to your MATLAB path by running:
   ```matlab
   addpath(genpath(pwd));
   savepath;
   ```
4. You are now ready to run the lecture scripts and project templates located in the `Lecture Scripts` and `Projects` folders.

**Code**:

All code used in the lectures (including the histogram estimation, Gaussian intensity modeling, and mask-based training scripts) will be available in this GitHub repository in the `Lecture Scripts` folder. 

## Copyright and License

No part of this series may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the authors, except in the case of brief quotations embodied in critical reviews and certain other noncommercial uses permitted by copyright law.
For permission requests, contact the authors.

The code provided in this series is for educational purposes only and should be used with caution. The authors are not responsible for any misuse of the code provided.

## Citations and Acknowledgments

If you find this series helpful and use it in your research or projects, please consider citing it as:

```bibtex
@software{Balaha_BE_640_Medical_2026,
  author  = {Balaha, Hossam Magdy and El-Baz, Ayman},
  month   = aug,
  title   = {{BE 640 Medical Image Analysis (Fall 2026)}},
  url     = {https://github.com/HossamBalaha/BE-640-Medical-Image-Analysis},
  version = {1.0},
  year    = {2026}
}
```

## Contact

This series is prepared and presented by `Hossam Magdy Balaha` from the University of Louisville's J.B. Speed School of
Engineering.

For any questions or inquiries, please contact me using the contact information available on my CV at the following
link: https://hossambalaha.github.io/
```