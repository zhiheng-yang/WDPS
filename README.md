# WDPS
To run the program from file input. You can simply input
```
python file_input.py input.txt
```
The program will generate outputs as
```
Question-001    Is Managua the capital of Nicaragua?
Question-001    R"
▶ What is the capital of Nicaragua?
▶ What are the 10 largest cities in Nicaragua?
▶ What is the population of Nicaragua?
▶ What is the currency of Nicaragua?
The National Congress is composed of a Chamber of Deputies with fifty members. The president of this Chamber appoints the President Pro-Tempore from within his group, who may then serve as Acting President in the absence of the President. The President's duties are largely ceremonial; he has no veto power and can only be removed by impeachment.
Nicaragua is a unitary republican country that consists of eighteen departments divided into three geographical regions: Pacific, Caribbean and Atlantic. The largest department by population and size is Masaya located in Central Highland Zone region. It has an area of 591 square kilometres (228 sq mi) and a population of 306,704 inhabitants as per the census taken in 2005.
The official name of the capital city of Nicaragua is Managua, but it goes by the unofficial name "Tipitapa" which means "place to be loved"
Question-001    A"yes"
Question-001    E"Managua"<TAB>"https://en.wikipedia.org/wiki/Managua"
Question-001    E"Nicaragua"<TAB>"https://en.wikipedia.org/wiki/Nicaragua"
Question-001    E"The National Congress"<TAB>"https://en.wikipedia.org/wiki/National_Congress_(Sri_Lanka)"
Question-001    E"Chamber of Deputies"<TAB>"https://en.wikipedia.org/wiki/Chamber_of_Deputies"
Question-001    E"Pacific"<TAB>"https://en.wikipedia.org/wiki/Pacific_Ocean"
Question-001    E"Caribbean"<TAB>"https://en.wikipedia.org/wiki/Caribbean"
Question-001    E"Atlantic"<TAB>"https://en.wikipedia.org/wiki/Atlantic_Ocean"
Question-001    E"Masaya"<TAB>"https://en.wikipedia.org/wiki/Masaya_Department"
Question-001    E"Central Highland Zone"<TAB>"Entity not linked to a knowledge base"
Question-001    E"2005"<TAB>"Entity not linked to a knowledge base"
Question-001    E"Tipitapa"<TAB>"https://en.wikipedia.org/wiki/Tipitapa"
Question-001    C"correct"
```
You can modify input.txt to add more questions.

#
Dataset preparation

## Download the dataset

### CoNLL-2003

```shell

```

### TACRED

### TACREV

#### Dev Split

```bash
python scripts/apply_tacred_patch.py \
  --dataset-file <TACRED DIR>/dev.json \
  --patch-file ./patch/dev_patch.json \
  --output-file ./dataset/dev_rev.json
```
#### Test Split

```bash
python scripts/apply_tacred_patch.py \
  --dataset-file <TACRED DIR>/test.json \
  --patch-file ./patch/test_patch.json \
  --output-file ./dataset/test_rev.json
```

### BoolQ


### 