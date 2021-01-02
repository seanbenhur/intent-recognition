# Intent Recognition with Pytorch,Torchtext and Streamlit
## Business Perspective
> Everyday people use many kinds of personal assistance systems such as  Google Assistant,Siri,Alexa,etc.. in those systems people usually ask questions such as <b>"What is the weather in Bay Area!?","Add Wonder to mendes playlist",</b>etc...,these type of questions/commands will help in increasing the training data of those systems, and these data can be grouped into common categories such as <b>GetWeather,AddToPlaylist</b>,etc.. so in this project a Machine learning system is created to classify these intents of various users

## Description
This motivation of the project is to classify the various intents of  the users into seven categories using methods of Deep Learning, I have used various architectures such as <br>
<ol>
	<li>RNN</li>
	<li>Bidirectional LSTM</li>
	<li>Fastext</li>
	<li>TEXTCNN</li>
	<li>BERT</li>
</ol>

## Installation
If you want to test this on your own machine I would recommend you run it in a virtual environment or use Docker, as not to affect the rest of your files.

### Python venv

In Python3 you can set up a virtual environment with

```bash
python3 -m venv /path/to/new/virtual/environment
```

Or by installing virtualenv with pip by doing
```bash
pip3 install virtualenv
```
Then creating the environment with
```bash
virtualenv venv
```
and finally activating it with
```bash
source venv/bin/activate
```
## Pretrained weights
> The pretrained weights of this project can be found [here](https://drive.google.com/drive/folders/1MBCxlfgghc0Buwndtw3JQ5kTpez88aeE?usp=sharing),which also contains the saved vocab file

## References
These repositiries highly helped me to build this project
<ul><li>[bentrevett/sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis)</li></ul>