'''
@author mojosaurus
This OM represents that document that will be passed around in the docproc pipeline
'''
import json

# Inputs to this class class be various, but it always returns a JSON object.
class Document:
    js = {}
    def __init__(self, text=""):
        self.js["original"] = text  # Keep the orignial text for reference
        self.js["text"] = text      # This is the filed that will be modified

    # Sets the value to key. Simple shit.
    def set(self, key, value):
        self.js[key] = value

    def get(self, key):
        return self.js[key]

    # Overload the __str__ function so that this can be used directly in print.
    def __str__(self):
        return  json.dumps(self.js, indent=1)

if __name__ == "__main__":
    doc = Document("Fellow world")
    print (doc)