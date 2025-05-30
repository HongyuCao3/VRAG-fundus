class ClassCombiner:
    def __init__(self):
        self.age_related = {
            "wet age-related macular degeneration": "age-related macular degeneration",
            "dry age-related macular degeneration": "age-related macular degeneration",
            "central retinal vein occlusion": "retinal vein occlusion",
            "branch retinal vein occlusion": "retinal vein occlusion",
            "central retinal artery occlusion": "retinal artery occlusion",
            "branch retinal artery occlusion": "retinal artery occlusion",
        }
        self.pcv = {
            "polypoidal choroidal vasculopathy\u00a0": "polypoidal choroidal vasculopathy"
        }

    def combine(self, ground_truth):
        if ground_truth in self.age_related.keys():
            return self.age_related[ground_truth]
        if ground_truth in self.pcv.keys():
            return self.pcv[ground_truth]
        return ground_truth
