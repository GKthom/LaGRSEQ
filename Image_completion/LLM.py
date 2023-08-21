import openai
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import numpy as np
from io import StringIO
openai.api_key  = os.environ["OPENAI_API_KEY"]

def get_completion(prompt,temp, model="gpt-4"):#change to gpt-3.5-turbo for GPT3
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def getLLMresponse(llminput,temp):
    text = f""" The following matrix corresponds to an image of size 10x10. 1 represents a white pixel, while 0 represents a black pixel.\
    If the initial image is:\
    """
    text=text+str(llminput)

    
    prompt=f"""Observe the pattern in the initial image. It is a partially complete pattern of a common geometric shape. Guess the final shape and display it in the form of a 10x10 matrix.\
    Note that all the initial pixels are in their correct positions. So the final image must necessarily contain the initial pixels as they are, and should not contain any additional pixels that are not part of the shape.\
    Make sure your response contains only the image, and not the explanation. Lets think step by step.\
    Example 1: If the initial image is: \
    [[1,0,0,0,0,0,0,0,1,0],
    [1,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0],
    [1,0,1,0,1,0,0,0,1,1]]\
    Then the full image may be in the shape of a square, in which case the final image would be:\
    [[1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1]]\
    Example 2: If the initial pixel configuration is: \
    [[0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0],
    [1,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [1,0,0,0,0,0,0,0,0,0],
    [1,1,1,0,1,1,1,0,1,1]]\
    Then the full image may be in the shape of a right angled triangle, such that the final pattern would be:\
    [[1,0,0,0,0,0,0,0,0,0],
    [1,0,1,0,0,0,0,0,0,0],
    [1,0,0,1,0,0,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0],
    [1,0,0,0,0,1,0,0,0,0],
    [1,0,0,0,0,0,1,0,0,0],
    [1,0,0,0,0,0,0,1,0,0],
    [1,0,0,0,0,0,0,0,1,0],
    [1,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1]]\
    Example 3: If the initial image is: \
    [[0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [1,0,0,1,0,1,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0]]\
    Then the full image may be depicting a horizontal line, such that the final pattern would be:\
    [[0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0]]\
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
    fin_ans=np.reshape(ans_arr,(10,10))#10,10 if size of image is 10x10
    return fin_ans

if __name__=="__main__":
    
    
    img=np.array([[0,0,0,0,0,0,0,0,0,1],
    [0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0]])
    print(getLLMresponse(img,0))
