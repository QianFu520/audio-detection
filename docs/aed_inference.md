{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;\f1\fmodern\fcharset0 Courier-Bold;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue255;\red255\green255\blue255;\red0\green0\blue0;
\red0\green0\blue109;\red144\green1\blue18;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c100000;\cssrgb\c100000\c100000\c100000;\cssrgb\c0\c0\c0;
\cssrgb\c0\c6275\c50196;\cssrgb\c63922\c8235\c8235;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 # TinyCNN Audio Inference Module\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ## Overview\cf0 \cb1 \strokec4 \
\
\cb3 This module is used to run inference with a trained TinyCNN model on a folder of audio clips. It can be used in two ways:\cb1 \
\
\cf2 \cb3 \strokec2 1. 
\f1\b \cf0 \strokec4 **As a Python module**
\f0\b0 , where other scripts can import the inference functions.\cb1 \
\cf2 \cb3 \strokec2 2. 
\f1\b \cf0 \strokec4 **As a standalone script**
\f0\b0 , where a folder of clips is processed and predictions are saved to a CSV file.\cb1 \
\
\cb3 Keeping the core inference functions separate from the command-line interface makes the code easier to reuse in future projects or analysis scripts.\cb1 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 # Module Structure\cf0 \cb1 \strokec4 \
\
\cb3 The module is divided into several main functions, each responsible for one part of the inference process.\cb1 \
\
\cf2 \cb3 \strokec2 ## `load_model()`\cf0 \cb1 \strokec4 \
\
\cb3 Loads a trained TinyCNN model from a checkpoint file and prepares it for inference.\cb1 \
\
\cb3 This function also reads saved model information, such as:\cb1 \
\
\cf2 \cb3 \strokec2 - \cf0 \strokec4 model version\cb1 \
\cf2 \cb3 \strokec2 - \cf0 \strokec4 training notes\cb1 \
\cf2 \cb3 \strokec2 - \cf0 \strokec4 evaluation metrics (if available)\cb1 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ## `preprocess_clip()`\cf0 \cb1 \strokec4 \
\
\cb3 Preprocesses a single audio clip into the format expected by the model.\cb1 \
\
\cb3 This ensures every clip is processed in the same way as during training.\cb1 \
\
\cb3 The preprocessing pipeline includes:\cb1 \
\
\cf2 \cb3 \strokec2 1. \cf0 \strokec4 Loading the audio using \cf5 \strokec5 `librosa`\cf0 \strokec4 .\cb1 \
\cf2 \cb3 \strokec2 2. \cf0 \strokec4 Resampling audio to 
\f1\b **48 kHz**
\f0\b0 .\cb1 \
\cf2 \cb3 \strokec2 3. \cf0 \strokec4 Converting audio to 
\f1\b **mono**
\f0\b0 .\cb1 \
\cf2 \cb3 \strokec2 4. \cf0 \strokec4 Keeping only the first 
\f1\b **3 seconds**
\f0\b0  of audio.\cb1 \
\cf2 \cb3 \strokec2 5. \cf0 \strokec4 Padding shorter clips with zeros.\cb1 \
\cf2 \cb3 \strokec2 6. \cf0 \strokec4 Generating a mel spectrogram with:\cb1 \
\cf2 \cb3 \strokec2    - \cf0 \strokec4 128 mel bands\cb1 \
\cf2 \cb3 \strokec2    - \cf0 \strokec4 FFT size (\cf5 \strokec5 `n_fft`\cf0 \strokec4 ) = 1024\cb1 \
\cf2 \cb3 \strokec2    - \cf0 \strokec4 Hop length = 512\cb1 \
\cf2 \cb3 \strokec2    - \cf0 \strokec4 Frequency range = 50\'9616,000 Hz\cb1 \
\cf2 \cb3 \strokec2 7. \cf0 \strokec4 Converting the spectrogram to the decibel scale.\cb1 \
\cf2 \cb3 \strokec2 8. \cf0 \strokec4 Keeping the first 256 time frames.\cb1 \
\cf2 \cb3 \strokec2 9. \cf0 \strokec4 Converting the data to \cf5 \strokec5 `float32`\cf0 \strokec4 .\cb1 \
\cf2 \cb3 \strokec2 10. \cf0 \strokec4 Reshaping into the format expected by TinyCNN.\cb1 \
\
\cb3 The final input tensor has the shape:\cb1 \
\
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 (B, 1, 128, 256)\cf0 \cb1 \strokec4 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cb3 where:\cb1 \
\
\cf2 \cb3 \strokec2 - \cf5 \strokec5 `B`\cf0 \strokec4  = batch size\cb1 \
\cf2 \cb3 \strokec2 - \cf5 \strokec5 `1`\cf0 \strokec4  = audio channel\cb1 \
\cf2 \cb3 \strokec2 - \cf5 \strokec5 `128`\cf0 \strokec4  = mel frequency bins\cb1 \
\cf2 \cb3 \strokec2 - \cf5 \strokec5 `256`\cf0 \strokec4  = time frames\cb1 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ## `predict()`\cf0 \cb1 \strokec4 \
\
\cb3 Runs the TinyCNN model on one or more preprocessed audio clips.\cb1 \
\
\cb3 Returns:\cb1 \
\
\cf2 \cb3 \strokec2 - \cf0 \strokec4 prediction probabilities from the model\cb1 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ## `score_folder()`\cf0 \cb1 \strokec4 \
\
\cb3 This is the main inference function of the module.\cb1 \
\
\cb3 It:\cb1 \
\
\cf2 \cb3 \strokec2 1. \cf0 \strokec4 Finds supported audio files in a folder.\cb1 \
\cf2 \cb3 \strokec2 2. \cf0 \strokec4 Preprocesses each clip.\cb1 \
\cf2 \cb3 \strokec2 3. \cf0 \strokec4 Runs inference in batches.\cb1 \
\cf2 \cb3 \strokec2 4. \cf0 \strokec4 Converts prediction probabilities into labels.\cb1 \
\cf2 \cb3 \strokec2 5. \cf0 \strokec4 Saves predictions to a CSV file.\cb1 \
\cf2 \cb3 \strokec2 6. \cf0 \strokec4 Returns predictions as a pandas DataFrame.\cb1 \
\
\cb3 Because \cf5 \strokec5 `score_folder()`\cf0 \strokec4  returns a DataFrame, other scripts can directly use the prediction results without needing to read the CSV file again.\cb1 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 # Inference Workflow\cf0 \cb1 \strokec4 \
\
\cb3 The complete inference workflow is:\cb1 \
\
\cf2 \cb3 \strokec2 1. \cf0 \strokec4 Load the trained TinyCNN model.\cb1 \
\cf2 \cb3 \strokec2 2. \cf0 \strokec4 Search the input folder for audio clips.\cb1 \
\cf2 \cb3 \strokec2 3. \cf0 \strokec4 Preprocess each audio clip.\cb1 \
\cf2 \cb3 \strokec2 4. \cf0 \strokec4 Run inference on batches of clips.\cb1 \
\cf2 \cb3 \strokec2 5. \cf0 \strokec4 Convert probabilities into predicted labels using a threshold.\cb1 \
\cf2 \cb3 \strokec2 6. \cf0 \strokec4 Save the prediction results to a CSV file.\cb1 \
\
\cb3 The default classification threshold is:\cb1 \
\
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 0.5\cf0 \cb1 \strokec4 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cb3 However, the threshold can be adjusted depending on the desired precision/recall tradeoff.\cb1 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 # Output\cf0 \cb1 \strokec4 \
\
\cb3 The output CSV contains one row for each processed audio clip.\cb1 \
\
\cf2 \cb3 \strokec2 | Column | Description |\cf0 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 |---|---|\cf0 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 |\cf0 \strokec4  \cf5 \strokec5 `clip_name`\cf0 \strokec4  \cf2 \strokec2 |\cf0 \strokec4  Name of the audio clip \cf2 \strokec2 |\cf0 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 |\cf0 \strokec4  \cf5 \strokec5 `audio_path`\cf0 \strokec4  \cf2 \strokec2 |\cf0 \strokec4  Path to the original audio file \cf2 \strokec2 |\cf0 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 |\cf0 \strokec4  \cf5 \strokec5 `not_meaningful_prob`\cf0 \strokec4  \cf2 \strokec2 |\cf0 \strokec4  Model prediction probability \cf2 \strokec2 |\cf0 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 |\cf0 \strokec4  \cf5 \strokec5 `predicted_label`\cf0 \strokec4  \cf2 \strokec2 |\cf0 \strokec4  Predicted class after applying the threshold \cf2 \strokec2 |\cf0 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 |\cf0 \strokec4  \cf5 \strokec5 `threshold`\cf0 \strokec4  \cf2 \strokec2 |\cf0 \strokec4  Threshold used for classification \cf2 \strokec2 |\cf0 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 |\cf0 \strokec4  \cf5 \strokec5 `model_version`\cf0 \strokec4  \cf2 \strokec2 |\cf0 \strokec4  Model version or checkpoint information \cf2 \strokec2 |\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 # Running the Module\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ## Option 1: Use as a Python Module\cf0 \cb1 \strokec4 \
\
\cb3 The inference function can be imported into another Python script.\cb1 \
\
\cf6 \cb3 \strokec6 ```python\cf0 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 from\cf0 \strokec4  ae_inference_callable \cf2 \strokec2 import\cf0 \strokec4  score_folder\cb1 \
\
\cb3 results = score_folder(\cb1 \
\cb3     folder_path=\cf6 \strokec6 "audio/"\cf0 \strokec4 ,\cb1 \
\cb3     model_path=\cf6 \strokec6 "outputs/models/tinycnn_v2.pth"\cf0 \strokec4 ,\cb1 \
\cb3     output_csv=\cf6 \strokec6 "outputs/predictions.csv"\cf0 \cb1 \strokec4 \
\cb3 )\cb1 \
\
\cf2 \cb3 \strokec2 print\cf0 \strokec4 (results.head())\cb1 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cb3 The returned object is a pandas DataFrame containing all predictions.\cb1 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ## Option 2: Run from the Command Line\cf0 \cb1 \strokec4 \
\
\cb3 Basic usage:\cb1 \
\
\cf6 \cb3 \strokec6 ```bash\cf0 \cb1 \strokec4 \
\cb3 python ae_inference_callable.py \\\cb1 \
\cb3     --input_folder audio/ \\\cb1 \
\cb3     --model_path outputs/models/tinycnn_v2.pth \\\cb1 \
\cb3     --output_csv outputs/predictions.csv\cb1 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 # Command-Line Arguments\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ## `--input_folder`\cf0 \cb1 \strokec4 \
\
\cb3 Path to the folder containing audio clips.\cb1 \
\
\cb3 Example:\cb1 \
\
\cf6 \cb3 \strokec6 ```bash\cf0 \cb1 \strokec4 \
\cb3 --input_folder unknown_clips/\cb1 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cb3 By default, the script searches recursively through all subfolders.\cb1 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ## `--model_path`\cf0 \cb1 \strokec4 \
\
\cb3 Path to the trained TinyCNN checkpoint.\cb1 \
\
\cb3 Example:\cb1 \
\
\cf6 \cb3 \strokec6 ```bash\cf0 \cb1 \strokec4 \
\cb3 --model_path outputs/models/tinycnn_v2.pth\cb1 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ## `--output_csv`\cf0 \cb1 \strokec4 \
\
\cb3 Path where the prediction CSV will be saved.\cb1 \
\
\cb3 Example:\cb1 \
\
\cf6 \cb3 \strokec6 ```bash\cf0 \cb1 \strokec4 \
\cb3 --output_csv outputs/inference_results.csv\cb1 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ## `--batch_size`\cf0 \cb1 \strokec4 \
\
\cb3 Controls how many audio clips are processed at once.\cb1 \
\
\cb3 Default:\cb1 \
\
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 64\cf0 \cb1 \strokec4 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cb3 Example:\cb1 \
\
\cf6 \cb3 \strokec6 ```bash\cf0 \cb1 \strokec4 \
\cb3 --batch_size 128\cb1 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cb3 A larger batch size can improve inference speed but requires more memory.\cb1 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ## `--threshold`\cf0 \cb1 \strokec4 \
\
\cb3 Sets the probability cutoff for converting model scores into predicted labels.\cb1 \
\
\cb3 Default:\cb1 \
\
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 0.5\cf0 \cb1 \strokec4 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cb3 Example:\cb1 \
\
\cf6 \cb3 \strokec6 ```bash\cf0 \cb1 \strokec4 \
\cb3 --threshold 0.9\cb1 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cb3 Example:\cb1 \
\
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 Probability = 0.95\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 Threshold = 0.90\cf0 \cb1 \strokec4 \
\
\cf5 \cb3 \strokec5 Prediction = 1\cf0 \cb1 \strokec4 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 Probability = 0.60\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 Threshold = 0.90\cf0 \cb1 \strokec4 \
\
\cf5 \cb3 \strokec5 Prediction = 0\cf0 \cb1 \strokec4 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ## `--no_recursive`\cf0 \cb1 \strokec4 \
\
\cb3 By default, the script searches all subfolders inside the input folder.\cb1 \
\
\cb3 Example folder:\cb1 \
\
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 audio/\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 \uc0\u9500 \u9472 \u9472  clip1.wav\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 \uc0\u9500 \u9472 \u9472  clip2.wav\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 \uc0\u9492 \u9472 \u9472  recorder_A/\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5     \uc0\u9492 \u9472 \u9472  clip3.wav\cf0 \cb1 \strokec4 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cb3 Default behavior:\cb1 \
\
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 Processes clip1.wav\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 Processes clip2.wav\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 Processes clip3.wav\cf0 \cb1 \strokec4 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cb3 To only process files directly inside the input folder:\cb1 \
\
\cf6 \cb3 \strokec6 ```bash\cf0 \cb1 \strokec4 \
\cb3 --no_recursive\cb1 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cb3 Result:\cb1 \
\
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 Processes clip1.wav\cf0 \cb1 \strokec4 \
\cf5 \cb3 \strokec5 Processes clip2.wav\cf0 \cb1 \strokec4 \
\
\cf5 \cb3 \strokec5 Ignores recorder_A/clip3.wav\cf0 \cb1 \strokec4 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 # Example Full Command\cf0 \cb1 \strokec4 \
\
\cb3 Example running inference on an unknown audio dataset:\cb1 \
\
\cf6 \cb3 \strokec6 ```bash\cf0 \cb1 \strokec4 \
\cb3 python ae_inference_callable.py \\\cb1 \
\cb3     --input_folder data/unknown_clips \\\cb1 \
\cb3     --model_path outputs/models/tinycnn_v2.pth \\\cb1 \
\cb3     --output_csv outputs/inference_v2.csv \\\cb1 \
\cb3     --batch_size 128 \\\cb1 \
\cb3     --threshold 0.9\cb1 \
\cf6 \cb3 \strokec6 ```\cf0 \cb1 \strokec4 \
\
\cb3 This command:\cb1 \
\
\cf2 \cb3 \strokec2 - \cf0 \strokec4 loads TinyCNN v2,\cb1 \
\cf2 \cb3 \strokec2 - \cf0 \strokec4 processes all audio clips in \cf5 \strokec5 `data/unknown_clips`\cf0 \strokec4 ,\cb1 \
\cf2 \cb3 \strokec2 - \cf0 \strokec4 uses batches of 128 clips,\cb1 \
\cf2 \cb3 \strokec2 - \cf0 \strokec4 classifies clips using a 0.9 probability threshold,\cb1 \
\cf2 \cb3 \strokec2 - \cf0 \strokec4 saves predictions to \cf5 \strokec5 `outputs/inference_v2.csv`\cf0 \strokec4 .\cb1 \
\
\cf2 \cb3 \strokec2 ---\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 # Notes\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 - \cf0 \strokec4 The model checkpoint must match the preprocessing pipeline used during training.\cb1 \
\cf2 \cb3 \strokec2 - \cf0 \strokec4 Changing the threshold changes the precision/recall tradeoff.\cb1 \
\cf2 \cb3 \strokec2 - \cf0 \strokec4 For large datasets, increasing \cf5 \strokec5 `batch_size`\cf0 \strokec4  can reduce inference time if enough memory is available.\cb1 \
\cf2 \cb3 \strokec2 - \cf0 \strokec4 The CSV output can be used directly for downstream analysis, auditing, or additional labeling.\cb1 \
}