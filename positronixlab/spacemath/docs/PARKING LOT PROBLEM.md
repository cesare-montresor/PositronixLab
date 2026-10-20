I have to come up with a very good name for a concept, something like "The dining philosophers problem"
Here the example I want to name:



There are about 100 people a square, scattered randomly.

For 1000 turns a speaker asks a yes/no question from a pool of N>1 questions, randomly.

If the answer is YES, the person rise the hand.
The speaker pinpoints the center of YES people. 
The YES people take one step TOWARDS the center.
Everybody else takes a step AWAY from the center.

At the end of the process, people will be organized in space in groups which emerged as dynamic equilibrium of the system. 


Examples of pools of questions:
- example 1 (N=3)
  Question A: who has a car?
  Question B: who has a motorbike?
  Question C: who has a bicylcle? 
 
- example 2 (N=2)
  Question A: who has cats?
  Question B: who has dogs?

- example 3 (N=2)
  Question A: who has brothers?
  Question B: who has sisters?

- example 4 (N=3+2+2=7)
  Ask all the 7 questions above.
  ( questions don't need to be related )



So going for a or a N=2 (A, B) example, each person can only fall into 4 different category.
+-----+-----+----------------------------------------------+-----------+
|  A  |  B  | Results                                      | Action    |
+-----+-----+----------------------------------------------+-----------+
| YES | YES | forming a group, at the center of the square | pull-pull |
| YES | NO  | forming a group, on the side on the center   | pull-push |
| NO  | YES | forming a group, on the side on the center   | pull-push |
| NO  | NO  | scattered at the edges of the square         | push-push |
+-----+-----+----------------------------------------------+-----------+
