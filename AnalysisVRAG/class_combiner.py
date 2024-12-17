class ClassCombiner():
    def __init__(self):
        self.age_related = {"wet age-related macular degeneration": "age-related macular degeneration","dry age-related macular degeneration":  "age-related macular degeneration"}
    
    def combine(self, ground_truth):
        if ground_truth in self.age_related.keys():
            return self.age_related[ground_truth]
        return ground_truth