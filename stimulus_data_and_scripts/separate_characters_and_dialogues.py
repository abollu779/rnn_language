import re
import pdb
import nltk
from collections import Counter

# What does this script do?
# Uses parsed txt file of CHARACTER: DIALOGUE to generate separate characters and dialogues txt files
# OLD CODE: Not necessary to create these files anymore, but can still be useful in some cases.

##################################
# EXTRACT UNIQUE CHARACTER NAMES #
##################################
# subtitles_file = "subtitles_with_characters/hidden_figures_subtitles.txt" #To edit: subtitles of the studied movies
# character_set = set()
# with open(subtitles_file, 'r') as f:
#     for line in f.readlines():
#         line = line.replace('\n','').replace('\t','')
#         if '-->' in line or line.isdigit() or line == '':
#             continue
#         elif line.find(' - ') != -1:
#                 character = line[:line.find(' - ')]
#                 character_set.add(character)
# main_characters = list(character_set)

# Hardcoded main_characters for Hidden Figures here as those with 20 or more lines.
# Easy to modify script above to make this restriction based on character dialogue counts.
main_characters = ['KATHERINE', 'AL HARRISON', 'DOROTHY', 'MARY', 'PAUL STAFFORD', 'JOHN GLENN', 'VIVIAN MITCHELL', 'JIM JOHNSON', 'RUTH', 'MISSION CONTROL COMMANDER']
character_psuedonyms = {'STAFFORD': 'PAUL STAFFORD'}

####################################################
# CREATE SEPARATE CHARACTERS.TXT AND DIALOGUES.TXT #
####################################################
def formatDialogueData(movie_name):
    with open('screenplays/parsed_character_lines/{}.txt'.format(movie_name), 'r') as f:
        characters, dialogues = [], []
        for line in f.readlines():
            character, dialogue = line.split(":")[0], line.split(":")[1]
            
            if character in character_psuedonyms:
                character = character_psuedonyms[character]
            if character not in main_characters:
                character = 'OTHER'
            
            for sentence in nltk.tokenize.sent_tokenize(dialogue.strip()):
                characters.append(character)
                dialogues.append(sentence)

    with open('screenplays/separated_characters_and_dialogues/{}_characters.txt'.format(movie_name), 'w+') as g:
        for character in characters:
            g.write("{}\r\n".format(character))
    with open('screenplays/separated_characters_and_dialogues/{}_dialogues.txt'.format(movie_name), 'w+') as h:
        for dialogue in dialogues:
            h.write("{}\r\n".format(dialogue))

if __name__=='__main__':
    formatDialogueData('hiddenfigures')

            