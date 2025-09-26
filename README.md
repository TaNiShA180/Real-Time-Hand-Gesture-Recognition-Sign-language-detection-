# Real-Time Hand Gesture Recognition using MediaPipe

This project is a complete pipeline for creating a real-time hand gesture recognition system. It uses OpenCV to capture video, MediaPipe for accurate hand landmark detection, and a Scikit-learn RandomForestClassifier to recognize custom gestures.

**Please Note**: This current implementation is trained to recognize three specific hand gestures, which are mapped to the labels 'A', 'B', and 'L'. You can easily extend this by following the steps in the **Customization** section.



***
## Features

-   **Real-Time Detection**: Classifies hand gestures instantly from a live webcam feed.
-   **Customizable Gestures**: Easily collect data and train the model to recognize your own set of gestures.
-   **Simple & Modular**: The project is broken down into clear, sequential scripts:
    1.  Image Collection
    2.  Dataset Creation
    3.  Model Training
    4.  Real-Time Inference

***
## Technologies Used

-   **Python**
-   **OpenCV**: For camera access and image processing.
-   **MediaPipe**: For high-fidelity hand and finger tracking.
-   **Scikit-learn**: For training the machine learning model.
-   **NumPy**: For numerical operations.
-   **Pickle**: For saving the dataset and trained model.

***
## Getting Started

Follow these steps to get the project up and running on your local machine.

### Prerequisites

-   Python 3.8+
-   A webcam connected to your computer.

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/TaNiShA180/Real-Time-Hand-Gesture-Recognition-Sign-language-detection-
    cd Real-Time-Hand-Gesture-Recognition
    ```

2.  **Create a virtual environment (recommended):**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```sh
    pip install -r requirements.txt
    ```

***
##  How to Run the Project

The project pipeline is divided into four main steps. Run the scripts in the following order.

### Step 1: Collect Image Data

Run the `collect_imgs.py` script to capture images for your gestures. The script will create a `./data` directory with sub-folders for each gesture class.

-   The script is pre-configured to collect **100 images** for **3 different classes** (0, 1, and 2).
-   When prompted, show your hand gesture to the camera and press **"Q"** to start collecting images for that class.
-   The script will automatically proceed to the next class after collecting 100 images.

```sh
python collect_imgs.py
```

### Step 2: Create the Dataset from Images

After collecting the images, run `create_dataset.py`. This script processes all images in the `./data` folder, extracts hand landmarks using MediaPipe, and saves the processed features and labels into a `data.pickle` file.

```sh
python create_dataset.py
```

### Step 3: Train the Classifier Model

Now, train the machine learning model using the dataset you just created. Running `train_classifier.py` will train a RandomForestClassifier and save the trained model as `model.p`. The script will also print the model's accuracy on the test set.

```sh
python train_classifier.py
```

You should see an output like:
`99.5% of samples were classified correctly !`

### Step 4: Run Real-Time Inference

This is the final step! Run `inference_classifier.py` to start your webcam and see the real-time gesture recognition in action.

-   The script loads the `model.p` file you trained.
-   Show a trained hand gesture to the camera, and the model will draw a bounding box and predict the gesture's label ('A', 'B', or 'L' as pre-configured).

```sh
python inference_classifier.py
```

***
## Customization

To add more gestures:

1.  In `collect_imgs.py`, change `number_of_classes` to your desired number.
2.  In `inference_classifier.py`, update the `labels_dict` to map the new class numbers to their corresponding characters or words. For example, for 4 classes:
    ```python
    labels_dict = {0: 'A', 1: 'B', 2: 'L', 3: 'C'}
    ```
3.  Re-run the entire pipeline (Steps 1-4).

***
## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
