# SEO Meta Tag Analyzer

## 02 - Planning and Building an SEO Analyzer

### Prompt 1

```markdown
Help me create an interactive app that displays the SEO (meta) tags for any website in an interactive and visual way to check that they're properly implemented.

The app should fetch the HTML for a site, then provide feedback on SEO tags in accordance with best practices for SEO optimization.

The app should give google and Social media previews.
```

### Prompt 2

```markdown
Make my app fully responsive and mobile friendly and fix some of the alignment and padding issues with the buttons and elements. Specifically, fix the centering of the overall SEO score and remove the "/100"
```

## 03 - Implementing SEO Analysis Features

### Prompt 1

```markdown
Make the application more visual:

- Create summaries for each category of meta tag that you will display visually to the user, similar to the overall score
- Make the app overall more visuall and allow me to get a summary of the SEO results at a glance
- Overall, lets make the app more visual and user friendly for folks that might be new to SEO so we can get a high-level view of how well our page is implemented

Do not remove any functionality, just make it easier to see a high level summary or drill down into the details
```

### Prompt 2

```markdown
Can you make the website entry form such that https:// is automatically populated and the user doesn't have to enter any other info
```

# NPS

## 04 - Planning and Building a Voting App

### Prompt 1

```markdown
Help me build an interactive app for voting and ranking the best National Parks.

The app should allow users to vote on parks head to head, then calculate a ranking for the parks based on the chess ELO system.

The app should prominently display the matchup along with overall rankings and recent votes.

http://en.wikipedia.org/wiki/List_of_national_parks_of_the_United_States
```

Get the text content of the wikipedia page, also include a screenshot of a wireframe of how the app looks like.


### Prompt 2

```markdown
The parks data are listed on the wikipedia page inside a table in the html, please fetch the page, download it, and extract all the parks from the source. There should be 63. Each park has an image in the table--you should use the externally hosted image as park image in our app

http://en.wikipedia.org/wiki/List_of_national_parks_of_the_United_States
```


### Prompt 3

```markdown
The rankings currently aren't working. The recent votes flow through, but I don't see any updates for the scores
```

### Prompt 4

```markdown
Ok let's now make our storage persistent--store all data in a database so it persists across sessions and users.
```

### Prompt 5

```markdown
Our app currently uses parks data hardcoded in parks_data.json, we'd like to move this to a postgres database. You should analyze the structure of the data and create a schema for rapid import. Be sure to check the data types and perform all necessary migrations.
```

## Enhancing the National Parks Voting App

### Prompt 1

```markdown
Help me understand what frameworks we used for our database--how does it work and how are we managing it?
```

### Prompt 2

```markdown
What is an ORM framework and why would I use drizzle orm?
```
