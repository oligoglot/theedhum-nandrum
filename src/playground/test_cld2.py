import cld2
import linecache
import sys

fileName = "resources/data/tamil_train.tsv"
lineNum = 11106 # Russian
lineNum = 11046 # tamil
lineNum = 8423 # telugu
lineNum = 7922 # tamil
#lineNum = 7787 # telugu
#lineNum = 7607 # telugu
lineNum = 570 # kannada
lineNum = 611 # kannada
line = linecache.getline(fileName, lineNum)
text = line.split("\t")[0].strip().encode("utf-8")
print(len(text))

isReliable, textBytesFound, details, vectors = cld2.detect(text, returnVectors=True)

print('  reliable: %s' % (isReliable != 0))
print('  textBytes: %s' % textBytesFound)
print('  details: %s' % str(details))
i=0
for vector in vectors:
    print ("*************")
    #print (vector)
    print (details[i])
    start = vector[0]
    end = vector[1]
    print ("Start : {}, end : {}".format(start, start+end))
    print (vector)
    print (text[start:start+end].decode("utf-8"))
    i += 1
    print ("*************")