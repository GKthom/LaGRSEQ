import openai
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import numpy as np
from io import StringIO
openai.api_key  =os.environ["OPENAI_API_KEY"]

def get_completion(prompt,temp, model="gpt-4"):#change to gpt-3.5-turbo for GPT3
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def getLLMresponse(llminput,temp):
    text = f""" The following matrix corresponds to positions on a 5x5 table. 1 represents the presence of an object, while 0 represents an empty space on the table.\
    If the initial table configuration is:\
    """
    text=text+str(llminput)

    
    prompt=f"""Observe the pattern of objects in the initial configuration. It is a partially complete pattern of a common scheme in which objects would be arranged. Guess the final arrangement scheme and display it in the form of a 5x5 matrix.\
    Note that all the initial object positions are correct. So the final configuration must necessarily contain the initial object positions as they are.\
    Lets think step by step. Make sure your response contains only the configuration, and not the explanation.\
    Example 1: If the initial arrangement coonfiguration is: \
    [[0,0,0,0,0],
    [0,1,0,1,0],
    [0,0,0,0,0],
    [0,1,1,0,0],
    [0,0,0,0,0]]\
    Then the full arrangement may be in the shape of a square, in which case the final arrangement would be:\
    [[0,0,0,0,0],
    [0,1,1,1,0],
    [0,1,0,1,0],
    [0,1,1,1,0],
    [0,0,0,0,0]]\
    Example 2: If the initial arrangement is: \
    [[0,0,1,0,0],
    [0,1,0,0,0],
    [0,0,0,1,0],
    [0,0,1,0,0],
    [0,0,0,0,0]]\
    Then a possible arrangement could be in the shape of an oval, such that the final arrangement would be:\
    [[0,0,1,0,0],
    [0,1,0,1,0],
    [0,1,0,1,0],
    [0,0,1,0,0],
    [0,0,0,0,0]]\
    ```{text}```
    """

    response=get_completion(prompt,temp)
    return response

def conv_ans_to_arr(answer):
    answer=answer.replace('[','')
    answer=answer.replace(']','')
    answer=answer.replace('"','')
    answer=answer.replace(',','')
    answer=answer.replace(' ','')
    answer=(answer.translate({ord(i): None for i in '\''}))
    answer=list(answer)
    ans_arr=[]
    for i in range(len(answer)):
        curr_str=answer[i]
        if curr_str.isdigit():
            ans_arr.append(int(curr_str))

    ans_arr=np.array(ans_arr)
    fin_ans=np.reshape(ans_arr,(5,5))#5,5 if arrangement pattern is as such
    return fin_ans

if __name__=="__main__":
    img=np.array([[0,0,1,0,0],
    [0,1,0,1,0],
    [0,0,0,0,1],
    [0,1,0,0,0],
    [0,0,1,0,0]])

    img_mod=img.astype('int')
    answer = conv_ans_to_arr(getLLMresponse(img_mod,0))
    print(answer)
