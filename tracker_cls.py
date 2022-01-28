
class Object_Tracker():
    def __init__(self, objectID, positions):
        self.position = [positions]
        self.id = objectID
        self.e_count = 0 #down
        self.ex_count = 0 # up
        self.e_status = ""
        self.ex_status = ""

