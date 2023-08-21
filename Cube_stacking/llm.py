import openai
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np

_ = load_dotenv(find_dotenv())
openai.api_key  = os.environ["OPENAI_API_KEY"]

def strtonum(seq):
    numseq=[]
    for i in range(len(seq)):
        if seq[i]=='a':
            intseq=0
        elif seq[i]=='b':
            intseq=1
        elif seq[i]=='c':
            intseq=2
        elif seq[i]=='d':
            intseq=3
        elif seq[i]=='e':
            intseq=4
        elif seq[i]=='f':
            intseq=5
        elif seq[i]=='g':
            intseq=6
        elif seq[i]=='h':
            intseq=7
        elif seq[i]=='i':
            intseq=8
        elif seq[i]=='j':
            intseq=9
        elif seq[i]=='k':
            intseq=10
        elif seq[i]=='l':
            intseq=11
        elif seq[i]=='m':
            intseq=12
        elif seq[i]=='n':
            intseq=13
        elif seq[i]=='o':
            intseq=14
        elif seq[i]=='p':
            intseq=15
        elif seq[i]=='q':
            intseq=16
        elif seq[i]=='r':
            intseq=17
        elif seq[i]=='s':
            intseq=18
        elif seq[i]=='t':
            intseq=19
        elif seq[i]=='u':
            intseq=20
        elif seq[i]=='v':
            intseq=21
        elif seq[i]=='w':
            intseq=22
        elif seq[i]=='x':
            intseq=23
        elif seq[i]=='y':
            intseq=24
        elif seq[i]=='z':
            intseq=25
        numseq.append(intseq)
    return numseq


def numtostr(seq):
    numseq=[]
    for i in range(len(seq)):
        if seq[i]==0:
            intseq='a'
        elif seq[i]==1:
            intseq='b'
        elif seq[i]==2:
            intseq='c'
        elif seq[i]==3:
            intseq='d'
        elif seq[i]==4:
            intseq='e'
        elif seq[i]==5:
            intseq='f'
        elif seq[i]==6:
            intseq='g'
        elif seq[i]==7:
            intseq='h'
        elif seq[i]==8:
            intseq='i'
        elif seq[i]==9:
            intseq='j'
        elif seq[i]==10:
            intseq='k'
        elif seq[i]==11:
            intseq='l'
        elif seq[i]==12:
            intseq='m'
        elif seq[i]==13:
            intseq='n'
        elif seq[i]==14:
            intseq='o'
        elif seq[i]==15:
            intseq='p'
        elif seq[i]==16:
            intseq='q'
        elif seq[i]==17:
            intseq='r'
        elif seq[i]==18:
            intseq='s'
        elif seq[i]==19:
            intseq='t'
        elif seq[i]==20:
            intseq='u'
        elif seq[i]==21:
            intseq='v'
        elif seq[i]==22:
            intseq='w'
        elif seq[i]==23:
            intseq='x'
        elif seq[i]==24:
            intseq='y'
        elif seq[i]==25:
            intseq='z'
        numseq.append(intseq)
    return numseq


def get_completion(prompt, temp, model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp,
    )
    return response.choices[0].message["content"]


def getLLMresponse(llminput,temp):
    text = f"""
    A table contains the following objects: \

     - Red cube of edge length 5cm (represented by 'a')
     - Red cube of edge length 4cm (represented by 'b')
     - Red cube of edge length 3cm (represented by 'c')
     - Red cube of edge length 2cm (represented by 'd')
     - Blue cube of edge length 10cm (represented by 'e')
     - Blue cube of edge length 8cm (represented by 'f')
     - Blue cube of edge length 6cm (represented by 'g')
     - Blue cube of edge length 2cm (represented by 'h')


    A human is currently stacking some of the cubes in the following sequence (from bottom to top): \

    """
    text=text+str(llminput)
    prompt=f"""You are an organizing robot. Stack the remaining cubes in the pattern that human seems to be following. \
    The final stack should have all 8 cubes. Lets think step by step.\
    For example, if the stack is initially in the order (bottom to top)\
    ['a','d','b']
    Then, the explanation is that the human is likely first stacking the red cubes irrespective of their edge lengths, \
    followed by the blue cubes. So a possible order would be: (bottom to top):\
    ['a','d','b','c','f','e','g','h']
    Make sure your response contains only the order in the form of a list and not the explanation.
    ```{text}```
    """

    response = get_completion(prompt,temp)
    return response


def conv_ans_to_arr(answer):
    answer=answer.replace('[','')
    answer=answer.replace(']','')
    answer=answer.replace('"','')
    answer=answer.replace(',','')
    answer=answer.replace(' ','')
    answer=(answer.translate({ord(i): None for i in '\''}))
    intans=[]
    try:
        intans=strtonum(answer)
    except:
        print("Bad LLM output format")

    return intans

def convarrtollminput(arr):
    strarr='['
    for i in range(len(arr)):
        strarr=strarr+'\''
        strarr=strarr+str(arr[i])
        strarr=strarr+'\''
        if i<len(arr)-1:
            strarr=strarr+','
    strarr=strarr+']'
    return strarr


if __name__=="__main__":
    answer=conv_ans_to_arr(getLLMresponse(['e','f','g','a'],0.0))
    print(answer)
