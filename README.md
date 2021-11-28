> There have been many efforts in Question Answer-ing in multiple languages. To evaluate each modelwith the actual and predicted answers we have cre-ated an Evaluation Tool1. The tool provides manyfeatures  to  the  researcher  to  analyze  the  resultswith user-friendly UI/UX. We have used Flask2as a backend by synchronizing with the evaluat-ing script.   Our tool can list the best and worst-performing samples for further analysis and candisplay collective EM and F1 score for the inputsamples.

> [1] Hariom A. Pandya, Bhavik Ardeshna, Dr. Brijesh S. Bhatt [*Cascading Adaptors to Leverage English Data to Improve Performance ofQuestion Answering for Low-Resource Languages*]()

## Get a copy of source code

> **Clone the SQuAD-Analytics (from the `main` branch) and `cd ` into the directory.**

```sh
git clone -b https://github.com/Bhavik-Ardeshna/SQuAD-Analytics.git
cd SQuAD-Analytic
```

## Before Installation and Running Locally
> **Create uploads & images directory inside the SQuAD-Analytics directory**
```sh
mkdir uploads
mkdir images
mkdir analysis
```

## Installation and Running Locally
> **Step to Create Env**
```sh
#if using python3
python3 -m venv env
```
```sh
python -m venv env
```
> **Step to run Flask Backend**

```sh
#if using python3
pip3 install requirements.txt
python3 app.py
```
```sh
pip install requirements.txt
python app.py
```

> **Go to this URL**
- [localhost:5000](https://localhost:5000)


<br>
<p align="center">
  <img src="https://github.com/Bhavik-Ardeshna/Question-Answering-Analytic-Tool/blob/main/assests/EvalTool1.png" alt="1"  width="80%"/>
</p>
<br>


<br>
<p align="center">
  <img src="https://github.com/Bhavik-Ardeshna/Question-Answering-Analytic-Tool/blob/main/assests/EvalTool2.png" alt="2"  width="80%"/>
</p>
<br>


<br>
<p align="center">
  <img src="https://github.com/Bhavik-Ardeshna/Question-Answering-Analytic-Tool/blob/main/assests/EvalTool3.png" alt="3"  width="80%"/>
</p>
<br>

