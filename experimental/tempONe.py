import numpy
import pandas

import markovify
from nearest_model_approaches.naive_approach_script import Naive_Approach


if __name__ == '__main__':

    # Get raw text as string.
    with open("Data/soaps_all.txt") as f:
        text = f.read()


    text=text.replace("  \n",".")
    text=text[0:50000]
    # Build the model.
    text_model = markovify.Text(text)


    #vals = text_model.chain.model - To get frequency distribution
    # Print five randomly-generated sentences
    '''
    for i in range(5):
        print(text_model.make_sentence())

    # Print three randomly-generated sentences of no more than 140 characters
    for i in range(3):
        print(text_model.make_short_sentence(140))
    '''
    #select sentence
    #  \ \\\\\\\\\\\\\\\\ \
    #select N closest sentences
    #from these N - top seed with first word -> pick nextword from collection of hashed elements

    #i like icecream
    #i like milk
    #i like blah

    #lets just try recreating the sentence

    text=text.split(".")
    naive = Naive_Approach()
    naive.load_embedding("D:/dramabot/Data/soap_embedding.txt")
    #naive.load_embedding(sys.argv[2])
    naive.fit(text)  # Cute baby model :)
    print "OK"
    what=naive._get_closest("mom  i still have a lot of questions about my life "
                           " but there's one thing that i know for"
                            " sure  and that is  i love jenny  and i "
                            "want to spend the rest of my life with her",30)

    print what
    print "OKKK "
    b = what[1:9]

    dict_val={}
    for i in b:
        for j in i.split(" "):
            if (j in dict_val) == False:
                dict_val[j]=1
            else:
                dict_val[j]=dict_val[j]+1

    print "DONE"

    '''
    n-1
    mom I
    ->

    (mom,I) -> Top list of choices


    '''