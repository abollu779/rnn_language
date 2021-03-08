import docx
import re
import pdb

### Screenplay must meet the following requirements before running this script:
# 1. If screenplay is collected as PDF, open with Word and save as docx.
# 2. Remove first page (usually contains title, written by, etc.) so that it starts
#    off from the very first line we read.
# 3. Remove any underlined AND italic text (this tends to express some writing displayed
#    on the screen like a movie title/year/location but is also centered text along with dialogues
#    which makes it inconvenient for us).

# Important Note: Hardcoded some statements to deal with the substrings "NASA" and "IBM" as 
# they are the only non-character words that appear in uppercase and can trip up the script.
uppercase_noncharacter_words = ["NASA", "IBM"]
def check_uppercase_noncharacter_words(line):
    for word in uppercase_noncharacter_words:
        if word in line:
            return True
    return False

def readScreenplay(movie_name):
    screenplay_path = 'screenplays/rawdocs/{}.docx'.format(movie_name)
    doc = docx.Document(screenplay_path)
    content = ""
    for p in doc.paragraphs:
        # removes text within parentheses and any leading/trailing whitespace. also replaces multiple spaces with one space.
        text = re.sub(r"\([^()]*\)", "", p.text).strip()
        text = re.sub(r"\s\s+", " ", text)
        text = text.replace("?!", "?").replace("!?", "?").replace(":", "-")
        if (len(text) > 0) and (p.paragraph_format.left_indent is not None and p.paragraph_format.left_indent > 0):
            content += text + "\n"

    dialogues = []
    lines = content.split("\n")[:-1] # last newline is at the end of the entire script this is to remove last empty string
    l, n = 0, len(lines)
    while l < n:
        line = lines[l]
        regex_line = line
        if check_uppercase_noncharacter_words(line):
            # this is the only case we need to separately deal with
            for word in uppercase_noncharacter_words:
                regex_line = regex_line.replace(word, "")

        if re.search("[A-Z]{2,}\s+(?![A-Z][A-Z])", regex_line):
            # line contains character name and their speech
            name_endidx = re.search("[A-Z]{2,}\s+(?![A-Z][A-Z])", regex_line).end() - 1
            dialogues.append(line[:name_endidx].strip())
            dialogues.append(line[name_endidx:].strip())
            l += 1
        elif re.search("[a-z0-9]", regex_line) is None:
            # line only contains character name
            assert l+1 < n # l cannot be final line as character name is always followed by dialogue
            next_line = lines[l+1]
            dialogues.append(line.strip())
            dialogues.append(next_line.strip())
            l += 2
        else:
            # line contains continued speech from previous line
            dialogues[-1] += " " + line.strip()
            l += 1
    return dialogues

def saveDialogues(dialogues, movie_name):
    with open("screenplays/formattedtxts/{}.txt".format(movie_name), 'w+') as f:
        i, n = 0, len(dialogues)
        while i < n:
            assert i+1 < n
            character, dialogue = dialogues[i], dialogues[i+1]
            f.write("{}: {}\r\n".format(character, dialogue))
            i += 2
    return


if __name__=='__main__':
    movie_name = 'hiddenfigures'
    dialogues = readScreenplay(movie_name)
    saveDialogues(dialogues, movie_name)