# Graph Teaching Task - Interactive Demo

Interactive demos for the three experiments from the paper "Mentalizing and heuristics as distinct cognitive strategies in human teaching".

**Try the task online:** [Graph Teaching Task Demo](https://sharootonian.github.io/CognitiveStrategiesInTeaching/)

## About

In these experiments, participants play the role of a teacher helping a student navigate through a graph to maximize points. The teacher can reveal a single path to improve the student's performance on subsequent trials.


## Repository Structure

```
GraphTeachingTask_demo/
├── index.html                           # Experiment selector
├── tasks/
│   ├── GraphTeachingTask/               # Experiment 1
│   ├── GraphTeachingTask_congruency/    # Experiment 2
│   └── GraphTeachingTask_scaffolding/   # Experiment 3
└── shared/
    ├── js/                              # JavaScript libraries
    ├── css/                             # Stylesheets
    ├── img/                             # Images
    └── data/                            # Trial configurations (CSV)
```

- **jsPsych 7.1.2** - Behavioral experiment library
- **AnyChart** - Graph visualization
  - Note: The original experiments used a licensed version without watermarks. This public demo uses the free version with watermarks.
- **d3-fetch** - CSV data loading
