TAAM, a mathematical argumentation model implemented in Python
====




This is a Python implementation of TAAM, a mathematical argumentation model.

If you want detailed information, please see the original article.(TODO)


## Description

This is a Python implementation of the mathematical argumentation model.

This source code was written to investigate the relationship between the satisfaction rates of various constraints. This source code was also used for collecting data of real-time spended to determine if a model satisfies each constraint or not.


## Setup

First, please intall graphviz on your computer.([doc](https://pygraphviz.github.io/documentation/stable/install.html))
We have confirmed that drawing is possible with version 5.0.1.

```
sudo apt-get install graphviz graphviz-dev # ubuntu
```

Then, run the following commands.
We used python 3.10.0.

```
python -m venv venv # Python 3.10.0 or later version of that is required.
pip install -r requirements.txt
```

The following commands can be executed to check if it works properly.

```
pytest
```
<!-- 
[pygraphvizに関してこれに従うとインストールはできた](https://github.com/pygraphviz/pygraphviz/issues/398) -->

## Usage



You can see the implementation of TAAM in the /model_src directory.



The data and the source　code used for analysis are located in /data directory.


If you want to check examples of models that meet or do not meet each constraint, 
please see /constraints directory. In this direcoty, you can see
the visualizations of the models. You can check the models in detail by loading the model  using .dill files.


## How the data was collected

Data were collected with and without bit pattern in Interpretation.
It is the latter data we used in our paper.


### Without bit pattern representation.

The data was collected with 2 PC.

- PC1  MacBook Pro（16 inch、2021）

![](https://i.imgur.com/EZ7yGB5.png)


- PC2 MacBook Pro（14 inch、2021）

![](https://i.imgur.com/QmLjfmc.png)


The base code used for generating data is located in ./task/measure_time.py(branch data1). We changed the setting to check various models.

In PC1, the range of number of propositional variables considered is from 1 to 2.
We collected data of about 200,000 models.

```
        num_node = random.randint(1,10)
        num_edge : int = int((num_node ** 2) * random.random() * (random.random() ** 0.3))

        num_onode = random.randint(0,num_node)
        num_pnode = num_node - num_onode

        

        Aord_size = num_onode
        Themes_size = random.randint(1,4)
        limit_num_given_themes = random.randint(1,4)

        num_propvar = random.randint(1,2)

        limit_image_size = random.randint(1,4)
```


In PC2, the range of propositional variables considered is from 1 to 3. 
We collected data of about 200,000 models.
```
        num_node = random.randint(1,10)
        num_edge : int = int((num_node ** 2) * random.random() * (random.random() ** 0.3))

        num_onode = random.randint(0,num_node)
        num_pnode = num_node - num_onode

        

        Aord_size = num_onode
        Themes_size = random.randint(1,4)
        limit_num_given_themes = random.randint(1,4)

        num_propvar = random.randint(1,3)

        limit_image_size = random.randint(1,4)

```

See the source code or the csv files below for details.(branch data1)



The amount of data collected using PC1: 316466 cases  (mainPC_time_data1.csv,mainPC_time_data2.csv,mainPC_time_data3.csv)

The amount of data collected using PC2: 235556 cases (subPC_time_data1.csv,subPC_time_data2.csv,subPC_time_data3.csv)

Total amount of data collected: 552022 cases



### With bit pattern representaion.

We used PC2 to collect data.

The base code used for generating data is located in ./task/measure_time.py(branch data2). We changed the setting as shown below.

```
        num_node = random.randint(1,10)
        num_edge : int = int((num_node ** 2) * random.random() * (random.random() ** 0.3))

        num_onode = random.randint(0,num_node)
        num_pnode = num_node - num_onode

        

        Aord_size = num_onode
        Themes_size = random.randint(1,4)
        limit_num_given_themes = random.randint(1,4)

        num_propvar = random.randint(1,3)

        limit_image_size = random.randint(1,4)


        bit_representation_ratio = 0.5

```

See the source code or the csv files below for details.(branch data2)


Total amount of data collected: 118073 cases


## manual

TAAM implemented in ./model_src/ is briefly explained here to show how to use it.


### Initialization of TAAM(`TAAMModel`)




TAAM is implemented in the `TAAMModel` class, which consists of a typed graph, a complete Boolean algebra, and interpretation, each of which must be initialized in the following manner. After initializing each of them, use the constructor of `TAAMModel` to generate a model.

#### Initialization of Typed Graphs(`TypedGraph`)


Initialize a typed graph using the constructor of the `TypedGraph` class.A typed graph is automatically generated such that well-formedness is satisfied; to check whether or not well-informedness is satisfied, the `is_well_formed` method can be used.

```
Args:
    Aord_size (int, optional): the number of elements in Aord_size(Aord = {"0","1",... }). Defaults to 6.
    Themes_size (int, optional):the number of Themes (Themes = {"t0","t1",...}). Defaults to 10.
    num_pnode (int, optional): the number of pnodes. Defaults to 5.
    num_onode (int, optional): the number of onodes. Defaults to 3.
    num_edge (int, optional): the number of edges. Defaults to 10.
    limit_num_given_themes (int, optional): Maximum number of themes given to vertices and edges. Defaults to 3.
```

```
>>> typed_graph = TypedGraph(Aord_size=2,Themes_size=2,num_pnode=1,num_onode=1,num_edge=1,limit_num_given_themes=2)
>>> print(typed_graph)
DiGraph with 2 nodes and 1 edges
Aord:['0', '1']
Themes:['t0', 't1']
```
See implementation for details.


#### Initialization of boolean algebra (`BooleanAlgebra`)


Initialize a boolean algebra using the constructor of the `BooleanAlgebra` class.
```
Args:
    num_propvar (int, optional): the number of propositional variables in the boolean Algebra. Defaults to 3.
```

It is recommended that the number of propositional variables be no more than 3, as the program may take considerably longer time to execute.

```
>>> D = BooleanAlgebra(num_propvar=2)
>>> print(D)
[A0, A1]
```

See implementation for details.





#### Initialization of interpretations(`Interpretation`)


Initialize an interpretation using the constructor of the `Interpretation` class with an initialized typed graph and  a boolean algebra.

```
Args:
    typed_graph (TypedGraph): typed graph
    D (BooleanAlgebra): boolean algebra
    limit_image_size (int, optional): the maximum number of 
                                      elements in the output of Interpretation. 
                                      Defaults to 5.
```

```
>>> I = Interpretation(typed_graph,D)
>>> print(I)
{((), '0'): [A0 & A1,
             (A0 & A1) | (A0 & ~A1),
             (A0 & A1) | (A0 & ~A1) | (~A0 & ~A1),
             (A0 & A1) | (A0 & ~A1) | (A1 & ~A0) | (~A0 & ~A1)],
 ((), '1'): [(A0 & A1) | (A0 & ~A1)],
<snip>
 (('t0', 't1'), 'omega'): [(A0 & A1) | (A0 & ~A1),
                           False,
                           (A0 & A1) | (A0 & ~A1) | (~A0 & ~A1),
                           (A0 & A1) | (~A0 & ~A1),
                           (A0 & A1) | (A0 & ~A1) | (A1 & ~A0)],
<snip>
 (('t1',), 't1.c'): []}

```


See implementation for details.




### How to check if constraints are satisfied

After initializing the model, use the  `meet_{constraint name}` methods.
True(False) is returned if the model is (un)satisfied.


### Saving and Loading models

Use save_model and load_model (static method) implemented in `TAAMModel`.

```python=
model = TAAMModel()
model.save("save location")
model.load("path_to_dill_file")
```


### Visualization of the model

You can visualize the model by using `visualize` methods.

An example is below.

```python
model = TAAMModel()

model.typed_graph.visualize(title="example1")

model.D.visualize(title="example2")

# description can be used to add more information about the model.
model.visualize(
    description = f"{model.typed_graph}\n I:{model.I}",
    title="example3"
)
```


## Licence

MIT License
