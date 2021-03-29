"""
Generate a dataset to evaluate bias in sense embeddings
"""

import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def find_sense_id(word):
    """
    Print the sense ids of words.
    """
    synsets = wn.synsets(word)
    for x in synsets:
        print(x.lemmas()[0].key(), x.pos(), "===", x.definition())


def people_template():
    """
    Generate nationalities vs. languages sense related stereotypes
    """
    nationalities = ["Japanese", "Chinese", "English", "Arabic", "German",
                     "French", "Spanish", "Portuguese", "Norwegian", "Swedish", "Polish", "Romanian",
                     "Russian", "Egyptian", "Finnish", "Vietnamese"]
    people_pleasant = ["beautiful", "nice", "friendly", "kind", "good looking", 
                    "intelligent", "clever", "generous", "funny", "cute", "handsome", "innocent"]
    people_unpleasant = ["ugly", "arrogant", "unkind", "unfriendly", "uppish", "foolish", "stupid", "cunning", "silly"]

    language_pleasant = ["easy to learn", "beautiful", "elegant", "soft", "easy to understand", "easy to write"]
    language_unpleasant = ["difficult to learn", "ugly", "rough", "hash", "difficult to understand", "difficult to write"]

    colour_pleasant = ["beautiful", "elegant", "attractive", "nice", "smart"]
    colour_unpleasant = ["ugly", "bad", "silly", "awful", "horrid"]

    templates = [("$1 people are $2.\t[nationality, $4%1:18:00::, $3]", nationalities, people_pleasant, "anti"), 
                ("$1 people are $2.\t[nationality, $4%1:18:00::, $3]", nationalities, people_unpleasant, "stereo"),
                ("$1 people are $2.\t[ethnicity, black%1:18:00::, $3]", ["Black"], people_pleasant, "anti"), 
                ("$1 people are $2.\t[ethinicity, black%1:18:00::, $3]", ["Black"], people_unpleasant, "stereo"),
                ("$1 dress was $2.\t[colour, black%1:07:00::, $3]", ["Black"], colour_pleasant, "anti"),
                ("$1 dress was $2.\t[colour, black%1:07:00::, $3]", ["Black"], colour_unpleasant, "stereo"),
                ("$1 language is $2.\t[language, $4%1:10:00::, $3]", nationalities, language_pleasant, "anti"),
                ("$1 language is $2.\t[language, $4%1:10:00::, $3]", nationalities, language_unpleasant, "stereo")]
    inst = []
    count = 0
    for template in templates:
        pattern = template[0]
        targets = template[1]
        attributes = template[2]
        label = template[3]
        for target in targets:
            for attribute in attributes:
                txt = pattern.replace("$1", target)
                txt = txt.replace("$2", attribute)
                txt = txt.replace("$3", label)
                txt = txt.replace("$4", target.lower())
                inst.append(txt)
                count += 1
            inst.append("")
    print("Total number of racial bias examples = {0}".format(count))
    return inst

def gender_template():
    """
    Generate gender related examples. For each profession, we consider its male and female variants.
    We will then associate each variant with a pleasant or unpleasant attributes to generate stereo or antistereo 
    examples. We will consider the verb sense of the profession to create gender bias unrelated examples.
    """

    occupations = [("engineer", "engineer%1:18:00::", "engineer%2:31:01::"),
                   ("carpenter", "carpenter%1:18:00::", "carpenter%2:41:00::"), 
                   ("guide", "guide%1:18:00::", "guide%2:38:00::"),
                   ("mentor", "mentor%1:18:00::", "mentor%2:32:00::"),
                   ("judge", "judge%1:18:00::", "judge%2:31:02::"),
                   ("nurse", "nurse%1:18:00::", "nurse%2:29:00::")]
    
    people_pleasant = ["nice", "friendly", "kind", "talented", "strong", "hard working",
                    "intelligent", "clever", "generous", "innocent", "meticulous", "clean", "honest", "professional"]
    people_unpleasant = ["ugly", "arrogant", "unkind", "unfriendly", "uppish", "foolish", "stupid", "cunning", "silly",
                        "lazy", "careless", "weak", "clumsy", "dirty", "cunning", "unprofessional", "unclean", "dishonest"]
    
    templates = [("$GENDER$ was a $ATTRIBUTE$ $OCCUPATION$.\t[noun, $SENSE-ID$, $BIAS$]")]

    count = 0
    inst = []
    for (occupation, noun_sense_id, verb_sense_id) in occupations:
        for template in templates:
            for attribute in people_pleasant:
                txt = template.replace("$ATTRIBUTE$", attribute)
                txt = txt.replace("$OCCUPATION$", occupation)
                txt = txt.replace("$SENSE-ID$", noun_sense_id)
                for (gender, bias) in [("He", "stereo"), ("She", "anti")]:
                    pattern = txt.replace("$GENDER$", gender)
                    pattern = pattern.replace("$BIAS$", bias)
                    inst.append(pattern)
                    count += 1
                inst.append("")
            
            for attribute in people_unpleasant:
                txt = template.replace("$ATTRIBUTE$", attribute)
                txt = txt.replace("$OCCUPATION$", occupation)
                txt = txt.replace("$SENSE-ID$", noun_sense_id)
                for (gender, bias) in [("He", "anti"), ("She", "stereo")]:
                    pattern = txt.replace("$GENDER$", gender)
                    pattern = pattern.replace("$BIAS$", bias)
                    inst.append(pattern)
                    count += 1
                inst.append("")
        
        with open("./%s-template" % occupation) as F:
            for line in F:
                pattern = line.strip()
                pattern = pattern.replace("$SENSE-ID$", verb_sense_id)

                 # If we have hard coded gender in the template then we do not have a pair.
                if pattern.find("$GENDER$") == -1: 
                    inst.append(pattern.capitalize())
                    count += 1
                else:
                    for (gender, bias) in [("he", "stereo"), ("she", "anti")]:                     
                        txt = pattern.replace("$GENDER$", gender)
                        txt = txt.replace("$BIAS$", bias)
                        inst.append(txt.capitalize())
                        count += 1
                    inst.append("")
        
    
    print("Total number of gender examples = {0}".format(count))

    
    return inst


def write_to_file(instances, fname):
    """
    Write the instances to a file.
    """
    with open(fname, 'w') as F:
        for inst in instances:
            F.write("%s\n" % inst)

def debug():
    find_sense_id('black')


def main():
    #instances = people_template()
    #write_to_file(instances, "racial-bias.txt")

    instances = gender_template()
    write_to_file(instances, "output")



if __name__ == "__main__":
    #debug()
    main()
    
