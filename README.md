<br />
<p align="center">
    <h2>Speech Sentimental Analysis using BERT</h2>
</p>

## Getting Started

To start with, it is recommended to create a virtual environment using the following command.

```sh
    python -m venv speechsent
```

Next activate the virtual environment.

#### Linux/Unix

```sh
    source ./speechsent/Scripts/activate
```

#### Windows

```sh
    ./speechsent/Scripts/activate
```
Then install all the dependencies using the command

```sh
    pip install -r requirements.txt
```

## Getting the model weights

1. To download the model`s weights, you can access the [google drive link](https://drive.google.com/file/d/1rdf3_I-kI8KhUiTtc7Dr3LY4YpTy3FkM/view?usp=sharing).

2. Paste the file in the [model](/model) folder and proceed forward with the execution.

## Execution

Execute the model using the following command.

```sh
    python audio_sentiment.py \[/audio/filename\]
```

The \[/audio/filename\] is optional, as you will be asked to specify the filename within the program execution.
