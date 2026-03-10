# Feature Extraction

We’ll represent each document (human-generated or AI-generated) with a vector $\vec{x}$.

Here’s the list of features we want to look for:

***Group A***: Features related to language, grammar and style, computed as

$$
x_{A_i} =\frac{k}{\text{total number of words}}
$$

where $k \in \mathbb{N}$ is the raw count of that feature:

1. The total number of emdash.
2. The total number of emojis.
3. The total number of words that are being bolded.
4. The total number of title cases used in the document.
5. The total number of vertical lists (ordered, unordered, number, bullet, dash, etc) used.
6. The total number of curly quotation marks and apostrophes.
7. The total number of table usages.
8. The total number of “AI vocabularly”
9. The total number of negative parallelisms of the form: “Not just X, but also Y.”
10. The total number of transition words used to begin sentences (e.g., additionally, consequently, notably).
11. The total number of grammatical mistakes.
12. The total number of phrasal templates and placeholder texts of the sort “[fill in the blank]”?

***Group B***: Features related to content, generally computed as the conditional probability of feature F to exist in our document given context C.

$$
x_{B_i} = {P(F|C)} = \frac{P(C|F) \times P(F)}{P(C)}
$$

1. Rule of three: total count of examples that come in triple triple (in contrast with examples of one, two, four and more examples).
2. The total number of basic copulatives ("is"/"are" phrases as opposed to a more sophisticated ones: “*serves as a”* or ”*mark the”*)
3. Number of elegant variations (i.e., repeatedly use a different synonym or related term to avoid overuse vs. repeating using similar terms)

***Group C***: Binary features:

$$
x_{C_i} = \left\{\begin{array}{ll}
      1 & \text{feature found} \\
      0 & \text{otherwise} \\
\end{array}
\right.
$$

1. Is there any subject line?
2. Is there any communication intended for the user (e.g., I hope this helps, Of course!, Certainly!, You're absolutely right!, Would you like..., is there anything else, let me know, more detailed breakdown, here is a ...)?
3. Is there any knowledge-cutoff disclaimers and/or speculation about sources (e.g., as of [date], Up to my last training update, as of my last knowledge update, While specific details are limited/scarce..., not widely available/documented/disclosed, ...in the provided/available sources/search results..., based on available information ...)?
4. Is there a summary section at the end (marked with: In summary, In conclusion, Overall)?

---

## More detailed descriptions

***Group A***: Features related to language

1. The total number of emdash.
2. The total number of emojis.
3. The total number of words that are being bolded.
4. The total number of title cases used in the document.
5. The total number of vertical lists (ordered, unordered, number, bullet, dash, etc) used.
6. The total number of curly quotation marks and apostrophes.
7. The total number of table usages.
8. The total number of phrasal templates and placeholder texts of the sort “[fill in the blank]”?
9. The total number of grammatical mistakes.
10. The total number of “AI vocabularly”
11. The total number of transition words used to begin sentences (e.g., additionally, consequently, notably).

***Group B***: Features related to content, generally computed as the conditional probability of feature F to exist in our document given context C.

$$
x_{B_i} = {P(F|C)} = \frac{P(C|F) \times P(F)}{P(C)}
$$

1. The total number of negative parallelisms of the form: “Not just X, but also Y.”
2. Rule of three: total count of examples that come in triple triple (in contrast with examples of one, two, four and more examples).
3. The total number of basic copulatives ("is"/"are" phrases as opposed to a more sophisticated ones: “*serves as a”* or ”*mark the”*)
4. Number of elegant variations (i.e., repeatedly use a different synonym or related term to avoid overuse vs. repeating using similar terms)

***Group C***: Binary features:

$$
x_{C_i} = \left\{\begin{array}{ll}
      1 & \text{feature found} \\
      0 & \text{otherwise} \\
\end{array}
\right.
$$

1. Is there any subject line?
2. Is there any communication intended for the user (e.g., I hope this helps, Of course!, Certainly!, You're absolutely right!, Would you like..., is there anything else, let me know, more detailed breakdown, here is a ...)?
3. Is there any knowledge-cutoff disclaimers and/or speculation about sources (e.g., as of [date], Up to my last training update, as of my last knowledge update, While specific details are limited/scarce..., not widely available/documented/disclosed, ...in the provided/available sources/search results..., based on available information ...)?
4. Is there a summary section at the end (marked with: In summary, In conclusion, Overall)?
