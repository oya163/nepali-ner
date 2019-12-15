## v2 Dataset Nepali NER

1. CoNLL formatted BIO tagging scheme
2. Corrected mistakes in v1

Some details of corrections are:-
* Unannotated words were annotated
* Incorrect tags were corrected

| Tokens   | v1  | v2    |
|----------|-----|-------|
| काठमाडौँ | O   | B-LOC |
| राजा     | PER | O     |




* Added missing hyphens, commas where necessary

| v1         | v2            |
|------------|---------------|
| नेपाल भारत | नेपाल  - भारत |

___

| v1 tokens |  Tags   | v2 tokens  |  Tags  |
|---------|-----|---------|-------|
| सशस्त्र | ORG | सशस्त्र | B-ORG |
| प्रहरी  | O   | प्रहरी  | I-ORG |

___

| v1 tokens | tags | v2 tokens | tags  |
|-----------|------|-----------|-------|
| मधेसी     | O    | मधेसी     | B-ORG |
| जनअधिकार  | O    | जनअधिकार  | I-ORG |
| फोरम      | ORG  | फोरम      | I-ORG |
| तमलोपा    | ORG  | ,         | O     |
| र         | O    | तमलोपा    | B-ORG |
| सद्भावना  | ORG  | र         | O     |
|           |      | सद्भावना  | B-ORG |

___

| v1 tokens      | tags | v2 tokens      | tags  |
|----------------|------|----------------|-------|
| प्रजातान्त्रिक | O    | प्रजातान्त्रिक | B-ORG |
| राष्ट्रिय      | ORG  | राष्ट्रिय      | I-ORG |
| युवा           | ORG  | युवा           | I-ORG |
| संघ            | ORG  | संघ            | I-ORG |
| नेपाल          | LOC  | नेपाल          | I-ORG |
