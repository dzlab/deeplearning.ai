{{For|the open-source mobile application framework|React Native}}

{{Short description|JavaScript library for building user interfaces}}
{{Infobox software
| name                   = React
| logo                   = React_Logo_SVG.svg
| author                 = Jordan Walke
| developer              = [[Meta Platforms|Meta]] and community
| released               = {{Start date and age|2013|5|29}}<ref name="initialrelease">{{cite web|access-date=22 Oct 2018|first1=Tom|first2=Jordan|last1=Occhino|last2=Walke|title=JS Apps at Facebook|url=https://www.youtube.com/watch?v=GW0rj4sNH2w|website=YouTube}}</ref>
| latest release version =
{{wikidata|property|reference|edit|P348}}
| latest release date    = {{start date and age|
{{wikidata|qualifier|P348|P577}}
}}
| latest preview version =
| latest preview date    = <!-- {{Start date and age|2016|04|7}}<ref name="ghrelease"/> -->
| programming language   = [[JavaScript]]
| platform               = [[Web platform]]
| genre                  = [[JavaScript library]]
| license                = [[MIT License]]
}}

'''React''' (also known as '''React.js''' or '''ReactJS''') is a [[Free and open-source software|free and open-source]] [[Frontend and backend|front-end]] [[JavaScript library]]<ref name="react">{{Cite web |title=React - A JavaScript library for building user interfaces. |url=https://reactjs.org |url-status=live |access-date=7 April 2018 |website=reactjs.org |language=en-US |archive-date=April 8, 2018 |archive-url=https://web.archive.org/web/20180408084010/https://reactjs.org/ }}</ref><ref>{{Cite web |title=Chapter 1. What Is React? - What React Is and Why It Matters [Book] |url=https://www.oreilly.com/library/view/what-react-is/9781491996744/ch01.html |url-status=live |access-date=2023-05-06 |website=www.oreilly.com |language=en |archive-date=May 6, 2023 |archive-url=https://web.archive.org/web/20230506100446/https://www.oreilly.com/library/view/what-react-is/9781491996744/ch01.html }}</ref> for building [[User interface|user interfaces]] based on [[Component-based software engineering|components]]. It is maintained by [[Meta Platforms|Meta]] (formerly Facebook) and a community of individual developers and companies.<ref>{{cite web |last=Krill |first=Paul |date=May 15, 2014 |title=React: Making faster, smoother UIs for data-driven Web apps |url=https://www.infoworld.com/article/2608181/javascript/react--making-faster--smoother-uis-for-data-driven-web-apps.html |access-date=2021-02-23 |website=[[InfoWorld]]}}</ref><ref>{{cite web |last=Hemel |first=Zef |date=June 3, 2013 |title=Facebook's React JavaScript User Interfaces Library Receives Mixed Reviews |url=https://www.infoq.com/news/2013/06/facebook-react |url-status=live |access-date=2022-01-11 |website=infoq.com |language=en-US |archive-url=https://web.archive.org/web/20220526082114/https://www.infoq.com/news/2013/06/facebook-react/ |archive-date=May 26, 2022}}</ref><ref>{{cite web |last=Dawson |first=Chris |date=July 25, 2014 |title=JavaScript's History and How it Led To ReactJS |url=https://thenewstack.io/javascripts-history-and-how-it-led-to-reactjs/ |url-status=live |access-date=2020-07-19 |website=The New Stack |language=en-US |archive-url=https://web.archive.org/web/20200806190027/https://thenewstack.io/javascripts-history-and-how-it-led-to-reactjs/ |archive-date=Aug 6, 2020 }}</ref>

React can be used to develop [[single-page application|single-page]], mobile, or [[Server-side rendering|server-rendered]] applications with frameworks like [[Next.js]]. Because React is only concerned with the user interface and rendering components to the [[Document Object Model|DOM]], React applications often rely on [[JavaScript libraries|libraries]] for routing and other client-side functionality.{{sfn|Dere|2017}}{{sfn|Panchal|2022}} A key advantage of React is that it only rerenders those parts of the page that have changed, avoiding unnecessary rerendering of unchanged DOM elements.

==Basic usage==
The following is a rudimentary example of using React for the web, written in [[#JSX|JSX]] and JavaScript.

<syntaxhighlight lang="javascript">
import React from 'react';
import ReactDOM from 'react-dom/client';

/** A pure component that displays a message */
const Greeting = () => {
  return (
    <div className="hello-world">
      <h1>Hello, world!</h1>
    </div>
  );
};

/** The main app component */
const App = () => {
  return <Greeting />;
};

/** React is rendered to a root element in the HTML page */
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
</syntaxhighlight>

based on the [[HTML]] document below.

<syntaxhighlight lang="html">
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>React App</title>
</head>
<body>
  <noscript>You need to enable JavaScript to run this app.</noscript>
  <div id="root"></div>
</body>
</html>
</syntaxhighlight>

The <code>Greeting</code> function is a React component that displays [["Hello, World!" program|<nowiki>''</nowiki>Hello, world"]].

When displayed on a [[web browser]], the result will be a rendering of:
<syntaxhighlight lang="html">
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>React App</title>
</head>
<body>
  <noscript>You need to enable JavaScript to run this app.</noscript>
  <div id="root">
    <div class="hello-world">
      <h1>Hello, world!</h1>
    </div>
  </div>
</body>
</html>
</syntaxhighlight>

==Notable features==
===Declarative===
React adheres to the [[declarative programming]] [[Programming paradigm|paradigm]].{{sfn|Wieruch|2020}}{{rp|76}} Developers design views for each state of an application, and React updates and renders components when data changes. This is in contrast with [[imperative programming]].{{sfn|Schwarzmüller|2018}}
===Components===
React code is made of entities called [[Component-based software engineering|components]].{{sfn|Wieruch|2020}}{{rp|10-12}} These components are modular and reusable.{{sfn|Wieruch|2020}}{{rp|70}} React applications typically consist of many layers of components. The components are rendered to a root element in the [[Document Object Model|DOM]] using the React DOM library. When rendering a component, values are passed between components through ''props'' (short for "properties")''.'' Values internal to a component are called its ''state.''<ref>{{cite web |title=Components and Props |url=https://reactjs.org/docs/components-and-props.html#props-are-read-only |url-status=live |access-date=7 April 2018 |website=React |publisher=Facebook |archive-date=7 April 2018 |archive-url=https://web.archive.org/web/20180407120115/https://reactjs.org/docs/components-and-props.html}}</ref>

The two primary ways of declaring components in React are through function components and class components.{{sfn|Wieruch|2020}}{{rp|118}}{{sfn|Larsen|2021}}{{rp|10}}

<syntaxhighlight lang="javascript">
import React from "react";

/** A pure component that displays a message with the current count */
const CountDisplay = props => {
  // The count value is passed to this component as props
  const { count } = props;
  return (<div>The current count is {count}.</div>);
}

/** A component that displays a message that updates each time the button is clicked */
const Counter = () => {
  // The React useState Hook is used here to store and update the 
  // total number of times the button has been clicked.
  const [count, setCount] = React.useState(0); 
  return (
    <div className="counter">
      <CountDisplay count={count} />
      <button onClick={() => setCount(count + 1)}>Add one!</button>
    </div>
  );
};
</syntaxhighlight>

=== Function components ===
Function components are declared with a function (using JavaScript function syntax or an [[Anonymous function|arrow function expression]]) that accepts a single "props" argument and returns JSX. From React v16.8 onwards, function components can use state with the <code>useState</code> Hook.

<syntaxhighlight lang="js">
// Function syntax
function Greeter() {
  return <div>Hello World</div>;
}

// Arrow function expression
const Greeter = () => <div>Hello World</div>;
</syntaxhighlight>

=== React Hooks ===
On February 16, 2019, React 16.8 was released to the public, introducing React Hooks.<ref>{{cite web
|url=https://reactjs.org/docs/hooks-intro.html
|title=Introducing Hooks
|publisher=react.js
|access-date=2019-05-20
}}</ref> Hooks are functions that let developers "hook into" React state and lifecycle features from function components.<ref>{{Cite web|url=https://reactjs.org/docs/hooks-overview.html|title=Hooks at a Glance – React|website=reactjs.org|language=en|access-date=2019-08-08}}</ref> Notably, Hooks do not work inside classes — they let developers use more features of React without classes.<ref>{{Cite web|url=https://blog.soshace.com/what-the-heck-is-react-hooks/|title=What the Heck is React Hooks?|date=2020-01-16|website=Soshace|language=en|access-date=2020-01-24}}</ref>

React provides several built-in Hooks such as <code>useState</code>,<ref>{{Cite web|url=https://reactjs.org/docs/hooks-state.html|title=Using the State Hook – React|website=reactjs.org|language=en|access-date=2020-01-24}}</ref>{{sfn|Larsen|2021}}{{rp|37}} <code>useContext</code>,{{sfn|Wieruch|2020}}{{rp|11}}<ref name=Larsen>{{Cite web|url=https://reactjs.org/docs/hooks-state.html|title=Using the State Hook – React|website=reactjs.org|language=en|access-date=2020-01-24}}</ref>{{sfn|Larsen|2021}}{{rp|12}} <code>useReducer</code>,{{sfn|Wieruch|2020}}{{rp|92}}<ref name=Larsen>{{Cite web|url=https://reactjs.org/docs/hooks-state.html|title=Using the State Hook – React|website=reactjs.org|language=en|access-date=2020-01-24}}</ref>{{sfn|Larsen|2021}}{{rp|65-66}} <code>useMemo</code>{{sfn|Wieruch|2020}}{{rp|154}}<ref name=Larsen>{{Cite web|url=https://reactjs.org/docs/hooks-state.html|title=Using the State Hook – React|website=reactjs.org|language=en|access-date=2020-01-24}}</ref>{{sfn|Larsen|2021}}{{rp|162}} and <code>useEffect</code>.<ref>{{Cite web|url=https://reactjs.org/docs/hooks-effect.html|title=Using the Effect Hook – React|website=reactjs.org|language=en|access-date=2020-01-24}}</ref>{{sfn|Larsen|2021}}{{rp|93-95}} Others are documented in the Hooks API Reference.<ref>{{Cite web|url=https://reactjs.org/docs/hooks-reference.html|title=Hooks API Reference – React|website=reactjs.org|language=en|access-date=2020-01-24}}</ref>{{sfn|Wieruch|2020}}{{rp|62}} <code>useState</code> and <code>useEffect</code>, which are the most commonly used, are for controlling state{{sfn|Wieruch|2020}}{{rp|37}} and side effects{{sfn|Wieruch|2020}}{{rp|61}} respectively.

==== Rules of hooks ====
There are two rules of Hooks<ref>{{Cite web|url=https://reactjs.org/docs/hooks-rules.html|title=Rules of Hooks – React|website=reactjs.org|language=en|access-date=2020-01-24}}</ref> which describe the characteristic code patterns that Hooks rely on:

# "Only Call Hooks at the Top Level" — Don't call hooks from inside loops, conditions, or nested statements so that the hooks are called in the same order each render.
# "Only Call Hooks from React Functions" — Don't call hooks from plain JavaScript functions so that stateful logic stays with the component.

Although these rules can't be enforced at runtime, code analysis tools such as [[Lint (software)|linters]] can be configured to detect many mistakes during development. The rules apply to both usage of Hooks and the implementation of custom Hooks,<ref>{{Cite web|url=https://reactjs.org/docs/hooks-custom.html|title=Building Your Own Hooks – React|website=reactjs.org|language=en|access-date=2020-01-24}}</ref> which may call other Hooks.

=== Server components ===
React server components or "RSC"s<ref>{{Cite web|url=https://react.dev/blog/2023/03/22/react-labs-what-we-have-been-working-on-march-2023#react-server-components|title=React Labs: What We've Been Working On – March 2023|website=react.dev|language=en|access-date=2023-07-23}}</ref> are function components that run exclusively on the server.  The concept was first introduced in the talk [https://react.dev/blog/2020/12/21/data-fetching-with-react-server-components Data Fetching with Server Components] Though a similar concept to Server Side Rendering, RSCs do not send corresponding JavaScript to the client as no hydration occurs.  As a result, they have no access to hooks. However, they may be [[Async/await| asynchronous function]], allowing them to directly perform asynchronous operations:

<syntaxhighlight lang="js" line="1">
async function MyComponent() {
  const message = await fetchMessageFromDb();

  return (
    <div>Message: {message}</div>
  );
}
</syntaxhighlight>

Currently, server components are most readily usable with [https://nextjs.org/docs/getting-started/react-essentials Next.js]. 

=== Class components ===
Class components are declared using [[ECMAScript|ES6]] classes. They behave the same way that function components do, but instead of using Hooks to manage state and lifecycle events, they use the lifecycle methods on the <code>React.Component</code> [[Inheritance (object-oriented programming)|base class]].
<syntaxhighlight lang="js" line="1">
class ParentComponent extends React.Component {
  state = { color: 'green' };
  render() {
    return (
      <ChildComponent color={this.state.color} />
    );
  }
}
</syntaxhighlight>The introduction of React Hooks with React 16.8 in February 2019 allowed developers to manage state and lifecycle behaviors within functional components, reducing the reliance on class components.

React Hooks, such as <code>useState</code> for state management and <code>useEffect</code> for side effects, have provided a more streamlined and concise way to build and manage React applications. This shift has led to improved code readability and reusability, encouraging developers to migrate from class components to functional components.

This trend aligns with the broader industry movement towards functional programming and modular design. As React continues to evolve, it's essential for developers to consider the benefits of functional components and React Hooks when building new applications or refactoring existing ones.<ref>{{Cite web |last=Chourasia |first=Rawnak |date=2023-03-08 |title=Convert Class Component to Function(Arrow) Component - React |url=https://codeparttime.com/convert-class-to-function-arrow-react/ |access-date=2023-08-15 |website=Code Part Time}}</ref>

=== Routing ===
React itself does not come with built-in support for [[routing]]. React is primarily a library for building user interfaces, and it doesn't include a full-fledged routing solution out of the box.

However, there are popular third-party libraries that can be used to handle routing in React applications. One such library is <code>react-router</code>, which provides a comprehensive routing solution for React applications.<ref>{{Cite web |date=2023-07-12 |title=Mastering React Router - The Ultimate Guide |url=https://www.devban.com/react-router-ultimate-guide/ |access-date=2023-07-26 |language=en-US}}</ref> It allows you to define routes, manage navigation, and handle URL changes in a React-friendly way.

To use <code>react-router</code>, you need to install it as a separate package and integrate it into your React application.

# Install <code>react-router-dom</code> using npm or yarn:<syntaxhighlight lang="bash">
npm install react-router-dom
</syntaxhighlight>
# Set up your routing configuration in your main application file:<syntaxhighlight lang="javascript">
import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';

import Home from './components/Home';
import About from './components/About';
import Contact from './components/Contact';

function App() {
  return (
    <Router>
      <Switch>
        <Route exact path="/" component={Home} />
        <Route path="/about" component={About} />
        <Route path="/contact" component={Contact} />
      </Switch>
    </Router>
  );
}

export default App;
</syntaxhighlight>
# Create the components for each route (e.g., Home, About, Contact).

With this setup, when the user navigates to different URLs, the corresponding components will be rendered based on the defined routes.[[File:VirtualDOM with respect to realDOM.png|thumb|There is a Virtual DOM that is used to implement the real DOM]]

===Virtual DOM===
Another notable feature is the use of a virtual [[Document Object Model]], or [[Virtual DOM]]. React creates an [[In-memory processing|in-memory]] data-structure cache, computes the resulting differences, and then updates the browser's displayed DOM efficiently.<ref name="workingwiththebrowser">{{cite web |title=Refs and the DOM |url=https://reactjs.org/docs/refs-and-the-dom.html |access-date=2021-07-19 |website=React Blog}}</ref> This process is called '''reconciliation'''. This allows the programmer to write code as if the entire page is rendered on each change, while React only renders the components that actually change. This selective rendering provides a major performance boost.<ref name=":0">{{Cite web |title=React: The Virtual DOM |url=https://www.codecademy.com/articles/react-virtual-dom |access-date=2021-10-14 |website=Codecademy |language=en}}</ref>

== Updates ==
When <code>ReactDOM.render</code><ref>{{Cite web |title=ReactDOM – React |url=https://reactjs.org/docs/react-dom.html |access-date=2023-01-08 |website=reactjs.org |language=en}}</ref> is called again for the same component and target, React represents the new UI state in the Virtual DOM and determines which parts (if any) of the living DOM needs to change.<ref>{{Cite web |title=Reconciliation – React |url=https://reactjs.org/docs/reconciliation.html |access-date=2023-01-08 |website=reactjs.org |language=en}}</ref>

[[File:React-example-virtual-dom-diff.svg|alt=Updates to realDOM are subject to virtualDOM|thumb|The virtualDOM will update the realDOM in real-time effortlessly]]

=== Lifecycle methods ===
Lifecycle methods for class-based components use a form of [[hooking]] that allows the execution of code at set points during a component's lifetime.

* <code>ShouldComponentUpdate</code> allows the developer to prevent unnecessary re-rendering of a component by returning false if a render is not required.
* <code>componentDidMount</code> is called once the component has "mounted" (the component has been created in the user interface, often by associating it with a [[Document Object Model|DOM]] node). This is commonly used to trigger data loading from a remote source via an [[API]].
*<code>componentWillUnmount</code> is called immediately before the component is torn down or "unmounted". This is commonly used to clear resource-demanding dependencies to the component that will not simply be removed with the unmounting of the component (e.g., removing any <code>setInterval()</code> instances that are related to the component, or an "[[Event (computing)|eventListener]]" set on the "document" because of the presence of the component)
* <code>render</code> is the most important lifecycle method and the only required one in any component. It is usually called every time the component's state is updated, which should be reflected in the user interface.

===JSX===
{{Main|JSX (JavaScript)|l1=JSX}}
[[JSX (JavaScript)|JSX]], or JavaScript Syntax Extension, is an extension to the JavaScript language syntax.<ref>{{cite web |date=2022-03-08 |title=Draft: JSX Specification |url=https://facebook.github.io/jsx/ |access-date=7 April 2018 |website=JSX |publisher=Facebook |language=en-US}}</ref> Similar in appearance to HTML,{{sfn|Wieruch|2020}}{{rp|11}} JSX provides a way to structure component rendering using syntax familiar{{sfn|Wieruch|2020}}{{rp|15}} to many developers. React components are typically written using JSX, although they do not have to be (components may also be written in pure JavaScript). JSX is similar to another extension syntax created by Facebook for [[PHP]] called [[XHP]].

An example of JSX code:
<syntaxhighlight lang="dart">
class App extends React.Component {
  render() {
    return (
      <div>
        <p>Header</p>
        <p>Content</p>
        <p>Footer</p>
      </div>
    );
  }
}
</syntaxhighlight>

===Architecture beyond HTML===
The basic [[Software architecture|architecture]] of React applies beyond rendering HTML in the browser. For example, Facebook has dynamic charts that render to <code><nowiki><canvas></nowiki></code> tags,<ref>{{cite web |last=Hunt |first=Pete |date=2013-06-05 |title=Why did we build React? – React Blog |url=https://facebook.github.io/react/blog/2013/06/05/why-react.html |access-date=2022-02-17 |website=reactjs.org |language=en-US}}</ref> and Netflix and [[PayPal]] use universal loading to render identical HTML on both the server and client.<ref name="paypal-isomorphic-reactjs">{{cite web |date=2015-04-27 |title=PayPal Isomorphic React |url=https://medium.com/paypal-engineering/isomorphic-react-apps-with-react-engine-17dae662379c |url-status=live |archive-url=https://web.archive.org/web/20190208124143/https://www.paypal-engineering.com/2015/04/27/isomorphic-react-apps-with-react-engine/ |archive-date=2019-02-08 |access-date=2019-02-08 |website=medium.com}}</ref><ref name="netflix-isomorphic-reactjs">{{cite web |date=2015-01-28 |title=Netflix Isomorphic React |url=http://techblog.netflix.com/2015/01/netflix-likes-react.html |access-date=2022-02-14 |website=netflixtechblog.com |language=en-US}}</ref>

=== Server-side Rendering ===
[[Server-side scripting|Server-side rendering]] (SSR) refers to the process of rendering a client-side JavaScript application on the server, rather than in the browser. This can improve the performance of the application, especially for users on slower connections or devices.

With SSR, the initial HTML that is sent to the client includes the fully rendered UI of the application. This allows the client's browser to display the UI immediately, rather than having to wait for the JavaScript to download and execute before rendering the UI.

React supports SSR, which allows developers to render React components on the server and send the resulting HTML to the client. This can be useful for improving the performance of the application, as well as for [[search engine optimization]] purposes.<syntaxhighlight lang="js" line="1">
const express = require('express');
const React = require('react');
const { renderToString } = require('react-dom/server');

const app = express();

app.get('/', (req, res) => {
  const html = renderToString(<MyApp />);
  res.send(`
    <!doctype html>
    <html>
      <body>
        <div id="root">${html}</div>
        <script src="/bundle.js"></script>
      </body>
    </html>
  `);
});

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});

</syntaxhighlight>

==Common idioms==
React does not attempt to provide a complete application library. It is designed specifically for building user interfaces<ref name="react" /> and therefore does not include many of the tools some developers might consider necessary to build an application. This allows the choice of whichever libraries the developer prefers to accomplish tasks such as performing network access or local data storage. Common patterns of usage have emerged as the library matures.

===Unidirectional data flow===
To support React's concept of unidirectional data flow (which might be contrasted with [[AngularJS]]'s bidirectional flow), the Flux architecture was developed as an alternative to the popular [[model–view–controller]] architecture. Flux features ''actions'' which are sent through a central ''dispatcher'' to a ''store'', and changes to the store are propagated back to the view.<ref name="flux">{{cite web|url=https://facebook.github.io/flux/docs/in-depth-overview|title=In Depth OverView|publisher=Facebook|access-date=7 April 2018|website=Flux}}</ref> When used with React, this propagation is accomplished through component properties. Since its conception, Flux has been superseded by libraries such as [[Redux (JavaScript library)|Redux]] and MobX.<ref>{{cite web|title=Flux Release 4.0|url=https://github.com/facebook/flux/releases/tag/4.0.0|website=Github|access-date=26 February 2021}}</ref>

Flux can be considered a variant of the [[observer pattern]].<ref>{{cite web|last1=Johnson|first1=Nicholas|title=Introduction to Flux - React Exercise|url=http://nicholasjohnson.com/react/course/exercises/flux/|website=Nicholas Johnson|access-date=7 April 2018}}</ref>

A React component under the Flux architecture should not directly modify any props passed to it, but should be passed [[callback function]]s that create ''actions'' which are sent by the dispatcher to modify the store. The action is an object whose responsibility is to describe what has taken place: for example, an action describing one user "following" another might contain a user id, a target user id, and the type <code>USER_FOLLOWED_ANOTHER_USER</code>.<ref>{{cite web|last1=Abramov|first1=Dan|title=The History of React and Flux with Dan Abramov|url=http://threedevsandamaybe.com/the-history-of-react-and-flux-with-dan-abramov/|website=Three Devs and a Maybe|access-date=7 April 2018}}</ref> The stores, which can be thought of as models, can alter themselves in response to actions received from the dispatcher.

This pattern is sometimes expressed as "properties flow down, actions flow up". Many implementations of Flux have been created since its inception, perhaps the most well-known being [[Redux (JavaScript library)|Redux]], which features a single store, often called a [[single source of truth]].<ref>{{cite web|title=State Management Tools - Results|url=http://2016.stateofjs.com/2016/statemanagement/|website=The State of JavaScript|access-date=29 October 2021}}</ref>

In February 2019, <code>useReducer</code> was introduced as a [[React (web framework)#React Hooks|React hook]] in the 16.8 release. It provides an API that is consistent with Redux, enabling developers to create Redux-like stores that are local to component states.<ref>[https://reactjs.org/blog/2019/02/06/react-v16.8.0.html#react-1 React v16.8: The One with Hooks]</ref>

==Future development==
Project status can be tracked via the core team discussion forum.<ref>{{Cite web|title = Meeting Notes|url = https://discuss.reactjs.org/c/meeting-notes|website = React Discuss|access-date = 2015-12-13}}</ref> However, major changes to React go through the Future of React repository issues and [[pull request]]s.<ref>{{Cite web|title = reactjs/react-future - The Future of React|url = https://github.com/reactjs/react-future|website = GitHub|access-date = 2015-12-13}}</ref><ref>{{Cite web|title = facebook/react - Feature request issues|url = https://github.com/facebook/react/labels/Type:%20Feature%20Request|website = GitHub|access-date = 2015-12-13}}</ref> This enables the React community to provide feedback on new potential features, experimental APIs and JavaScript syntax improvements.

==History==
React was created by Jordan Walke, a software engineer at [[Meta Platforms|Meta]], who initially developed a prototype called "F-Bolt" <ref name="How A Small Team of Developers Created React at Facebook | React.js: The Documentary">{{cite web |title=React.js: The Documentary |url=https://youtube.com/watch?v=8pDqJVdNa44?si=FMJqegC4dPtwKP__&t=528 |website=Youtube |publisher=Honeypot}}</ref> before later renaming it to "FaxJS". This early version is documented in Jordan Walke's GitHub repository.{{ref|Walke, Jordan. "FaxJS." GitHub. https://github.com/jordwalke/FaxJs. Accessed 11 July 2019.}} Influences for the project included [[XHP]], an [[HTML]] component library for [[PHP]].

React was first deployed on Facebook's [[News Feed]] in 2011 and subsequently integrated into [[Instagram]] in 2012. However, a specific citation is needed to verify this claim. In May 2013, at JSConf US, the project was officially open-sourced, marking a significant turning point in its adoption and growth.{{ref|Hámori, Emergent. "React - The Pragmatic Guide." 2022.}}


[[React Native]], which enables native [[Android (operating system)|Android]], [[iOS]], and [[Universal Windows Platform|UWP]] development with React, was announced at Facebook's React Conf in February 2015 and open-sourced in March 2015.

On April 18, 2017, Facebook announced React Fiber, a new set of internal algorithms for rendering, as opposed to React's old rendering algorithm, Stack.{{sfn|Lardinois|2017}} React Fiber was to become the foundation of any future improvements and feature development of the React library.<ref>{{cite web|title = React Fiber Architecture|url = https://github.com/acdlite/react-fiber-architecture| website=Github|access-date = 19 April 2017}}</ref>{{Update inline|reason=Last commit was in 2016. Is this statement still true?|date=June 2018}} The actual syntax for programming with React does not change; only the way that the syntax is executed has changed.<ref name="techcrunch">{{cite web|url=https://techcrunch.com/2017/04/18/facebook-announces-react-fiber-a-rewrite-of-its-react-framework/|title=Facebook announces React Fiber, a rewrite of its React framework|website=TechCrunch|accessdate=2018-10-19}}</ref> React's old rendering system, Stack, was developed at a time when the focus of the system on dynamic change was not understood. Stack was slow to draw complex animation, for example, trying to accomplish all of it in one chunk. Fiber breaks down animation into segments that can be spread out over multiple frames. Likewise, the structure of a page can be broken into segments that may be maintained and updated separately. JavaScript functions and virtual [[Document Object Model|DOM]] objects are called "fibers", and each can be operated and updated separately, allowing for smoother on-screen rendering.<ref name="github">{{cite web|url=https://github.com/acdlite/react-fiber-architecture|title=GitHub - acdlite/react-fiber-architecture: A description of React's new core algorithm, React Fiber|website=github.com|accessdate=2018-10-19}}</ref>

On September 26, 2017, React 16.0 was released to the public.<ref>{{cite web
|url=https://reactjs.org/blog/2017/09/26/react-v16.0.html
|title=React v16.0
|publisher=react.js
|date=2017-09-26
|access-date=2019-05-20
}}</ref>

On August 10, 2020, the React team announced the first release candidate for React v17.0, notable as the first major release without major changes to the React developer-facing API.<ref>url=https://reactjs.org/blog/2020/08/10/react-v17-rc.html</ref>

On March 29, 2022, React 18 was released which introduced a new concurrent renderer, automatic batching and support for server side rendering with Suspense.<ref name=":1" />

{| class="wikitable mw-collapsible mw-collapsed"
|+Versions
|- style="position:sticky; top:0"
!Version
!Release Date
!Changes
|-
|0.3.0
|29 May 2013
|Initial Public Release
|-
|0.4.0
|20 July 2013
|Support for comment nodes {{tag|div|content={{mset|/* */}}}}, Improved server-side rendering APIs, Removed React.autoBind, Support for the key prop, Improvements to forms, Fixed bugs.
|-
|0.5.0
|20 October 2013
|Improve Memory usage, Support for Selection and Composition events, Support for getInitialState and getDefaultProps in mixins, Added React.version and React.isValidClass, Improved compatibility for Windows.
|-
|0.8.0
|20 December 2013
|Added support for rows & cols, defer & async, loop for {{tag|audio|o}} & {{tag|video|o}}, autoCorrect attributes. Added onContextMenu events, Upgraded jstransform and esprima-fb tools, Upgraded browserify.
|-
|0.9.0
|20 February 2014
|Added support for crossOrigin, download and hrefLang, mediaGroup and muted, sandbox, seamless, and srcDoc, scope attributes, Added any, arrayOf, component, oneOfType, renderable, shape to React.PropTypes, Added support for onMouseOver and onMouseOut event, Added support for onLoad and onError on {{tag|img|o}} elements.
|-
|0.10.0
|21 March 2014
|Added support for srcSet and textAnchor attributes, add update function for immutable data, Ensure all void elements don't insert a closing tag.
|-
|0.11.0
|17 July 2014
|Improved SVG support, Normalized e.view event, Update $apply command, Added support for namespaces, Added new transformWithDetails API, includes pre-built packages under dist/, MyComponent() now returns a descriptor, not an instance.
|-
|0.12.0
|21 November 2014
|Added new features Spread operator ({...}) introduced to deprecate this.transferPropsTo, Added support for acceptCharset, classID, manifest HTML attributes, React.addons.batchedUpdates added to API, @jsx React.DOM no longer required, Fixed issues with CSS Transitions.
|-
|0.13.0
|10 March 2015
|Deprecated patterns that warned in 0.12 no longer work, ref resolution order has changed, Removed properties this._pendingState and this._rootNodeID, Support ES6 classes, Added API React.findDOMNode(component), Support for iterators and immutable-js sequences, Added new features React.addons.createFragment, deprecated React.addons.classSet.
|-
|0.14.1
|29 October 2015
|Added support for srcLang, default, kind attributes, and color attribute, Ensured legacy .props access on DOM nodes, Fixed scryRenderedDOMComponentsWithClass, Added react-dom.js.
|-
|15.0.0
|7 April 2016
|Initial render now uses document.createElement instead of generating HTML, No more extra {{tag|span|o}}s, Improved SVG support, {{code|ReactPerf.getLastMeasurements()}} is opaque, New deprecations introduced with a warning, Fixed multiple small memory leaks, React DOM now supports the cite and profile HTML attributes and cssFloat, gridRow and gridColumn CSS properties.
|-
|15.1.0
|20 May 2016
|Fix a batching bug, Ensure use of the latest object-assign, Fix regression, Remove use of merge utility, Renamed some modules.
|-
|15.2.0
|1 July 2016
|Include component stack information, Stop validating props at mount time, Add React.PropTypes.symbol, Add onLoad handling to {{tag|link|o}} and onError handling to {{tag|source|o}} element, Add {{code|isRunning()}} API, Fix performance regression.
|-
|15.3.0
|30 July 2016
|Add React.PureComponent, Fix issue with nested server rendering, Add xmlns, xmlnsXlink to support SVG attributes and referrerPolicy to HTML attributes, updates React Perf Add-on, Fixed issue with ref.
|-
|15.3.1
|19 August 2016
|Improve performance of development builds, Cleanup internal hooks, Upgrade fbjs, Improve startup time of React, Fix memory leak in server rendering, fix React Test Renderer, Change trackedTouchCount invariant into a console.error.
|-
|15.4.0
|16 November 2016
|React package and browser build no longer includes React DOM, Improved development performance, Fixed occasional test failures, update batchedUpdates API, React Perf, and {{code|ReactTestRenderer.create()}}.
|-
|15.4.1
|23 November 2016
|Restructure variable assignment, Fixed event handling, Fixed compatibility of browser build with AMD environments.
|-
|15.4.2
|6 January 2017
|Fixed build issues, Added missing package dependencies, Improved error messages.
|-
|15.5.0
|7 April 2017
|Added react-dom/test-utils, Removed peerDependencies, Fixed issue with Closure Compiler, Added a deprecation warning for React.createClass and React.PropTypes, Fixed Chrome bug.
|-
|15.5.4
|11 April 2017
|Fix compatibility with Enzyme by exposing batchedUpdates on shallow renderer, Update version of prop-types, Fix react-addons-create-fragment package to include loose-envify transform.
|-
|15.6.0
|13 June 2017
|Add support for CSS variables in style attribute and Grid style properties, Fix AMD support for addons depending on react, Remove unnecessary dependency, Add a deprecation warning for React.createClass and React.DOM factory helpers.
|-
|16.0.0
|26 September 2017
|Improved error handling with introduction of "error boundaries", React DOM allows passing non-standard attributes, Minor changes to setState behavior, remove react-with-addons.js build, Add React.createClass as create-react-class, React.PropTypes as prop-types, React.DOM as react-dom-factories, changes to the behavior of scheduling and lifecycle methods.
|-
|16.1.0
|9 November 2017
|Discontinuing Bower Releases, Fix an accidental extra global variable in the UMD builds, Fix onMouseEnter and onMouseLeave firing, Fix <textarea> placeholder, Remove unused code, Add a missing package.json dependency, Add support for React DevTools.
|-
|16.3.0
|29 March 2018
|Add a new officially supported context API, Add new packagePrevent an infinite loop when attempting to render portals with SSR, Fix an issue with this.state, Fix an IE/Edge issue.
|-
|16.3.1
|3 April 2018
|Prefix private API, Fix performance regression and error handling bugs in development mode, Add peer dependency, Fix a false positive warning in IE11 when using Fragment.
|-
|16.3.2
|16 April 2018
|Fix an IE crash, Fix labels in User Timing measurements, Add a UMD build, Improve performance of unstable_observedBits API with nesting.
|-
|16.4.0
|24 May 2018
|Add support for Pointer Events specification, Add the ability to specify propTypes, Fix reading context, Fix the {{code|getDerivedStateFromProps()}} support, Fix a testInstance.parent crash, Add React.unstable_Profiler component for measuring performance, Change internal event names.
|-
|16.5.0
|5 September 2018
|Add support for React DevTools Profiler, Handle errors in more edge cases gracefully, Add react-dom/profiling, Add onAuxClick event for browsers, Add movementX and movementY fields to mouse events, Add tangentialPressure and twist fields to pointer event.
|-
|16.6.0
|23 October 2018
|Add support for contextType, Support priority levels, continuations, and wrapped callbacks, Improve the fallback mechanism, Fix gray overlay on iOS Safari, Add {{code|React.lazy()}} for code splitting components.
|-
|16.7.0
|20 December 2018
|Fix performance of React.lazy for lazily-loaded components, Clear fields on unmount to avoid memory leaks, Fix bug with SSR, Fix a performance regression.
|-
|16.8.0
|6 February 2019
|Add Hooks, Add {{code|ReactTestRenderer.act()}} and {{code|ReactTestUtils.act()}} for batching updates, Support synchronous thenables passed to React.lazy(), Improve useReducer Hook lazy initialization API.
|-
|16.8.6
|27 March 2019
|Fix an incorrect bailout in useReducer(), Fix iframe warnings in Safari DevTools, Warn if contextType is set to Context.Consumer instead of Context, Warn if contextType is set to invalid values.
|-
|16.9.0
|9 August 2019
|Add {{mono|React.Profiler}} API for gathering performance measurements programmatically. Remove unstable_ConcurrentMode in favor of unstable_createRoot
|-
|16.10.0
|27 September 2019
|Fix edge case where a hook update wasn't being memoized. Fix heuristic for determining when to hydrate, so we don't incorrectly hydrate during an update. Clear additional fiber fields during unmount to save memory. Fix bug with required text fields in Firefox. Prefer Object.is instead of inline polyfill, when available. Fix bug when mixing Suspense and error handling.
|-
|16.10.1
|28 September 2019
|Fix regression in Next.js apps by allowing Suspense mismatch during hydration to silently proceed
|-
|16.10.2
|3 October 2019
|Fix regression in react-native-web by restoring order of arguments in event plugin extractors
|-
|16.11.0
|22 October 2019
|Fix mouseenter handlers from firing twice inside nested React containers. Remove unstable_createRoot and unstable_createSyncRoot experimental APIs. (These are available in the Experimental channel as createRoot and createSyncRoot.)
|-
|16.12.0
|14 November 2019
|React DOM - Fix passive effects (<code>useEffect</code>) not being fired in a multi-root app.

React Is - Fix <code>lazy</code> and <code>memo</code> types considered elements instead of components
|-
|16.13.0
|26 February 2020
|Features added in React Concurrent mode.
Fix regressions in React core library and React Dom.
|-
|16.13.1
|19 March 2020
|Fix bug in legacy mode Suspense.
Revert warning for cross-component updates that happen inside class render lifecycles
|-
|16.14.0
|14 October 2020
|Add support for the new JSX transform.
|-
|17.0.0
|20 October 2020
|"No New Features" enables gradual React updates from older versions.
Add new JSX Transform, Changes to Event Delegation
|-
|17.0.1
|22 October 2020
|React DOM - Fixes a crash in IE11
|-
|17.0.2
|22 March 2021
|React DOM - Remove an unused dependency to address the <code>SharedArrayBuffer</code> cross-origin isolation warning.
|-
|18.0.0
|29 March 2022
|Concurrent React, Automatic batching, New Suspense Features, Transitions, Client and Server Rendering APIs, New Strict Mode Behaviors, New Hooks <ref name=":1">{{cite web|title=React v18.0|url=https://reactjs.org/blog/2022/03/29/react-v18.html|website=reactjs.org|language=en|access-date=2022-04-12}}</ref>
|-
|18.1.0
|26 April 2022
|Many fixes and performance improvements
|-
|18.2.0
|14 June 2022
|Many more fixes and performance improvements
|}

==Licensing==
The initial public release of React in May 2013 used the [[Apache License 2.0]]. In October 2014, React 0.12.00 replaced this with the [[BSD licenses#3-clause|3-clause BSD license]] and added a separate PATENTS text file that permits usage of any Facebook patents related to the software:<ref>{{cite web|title=React CHANGELOG.md|url=https://github.com/facebook/react/blob/master/CHANGELOG.md#0120-october-28-2014|website=GitHub}}</ref><blockquote>The license granted hereunder will terminate, automatically and without notice, for anyone that makes any claim (including by filing any lawsuit, assertion or other action) alleging (a) direct, indirect, or contributory infringement or inducement to infringe any patent: (i) by Facebook or any of its subsidiaries or affiliates, whether or not such claim is related to the Software, (ii) by any party if such claim arises in whole or in part from any software, product or service of Facebook or any of its subsidiaries or affiliates, whether or not such claim is related to the Software, or (iii) by any party relating to the Software; or (b) that any right in any patent claim of Facebook is invalid or unenforceable.</blockquote>This unconventional clause caused some controversy and debate in the React user community, because it could be interpreted to empower Facebook to revoke the license in many scenarios, for example, if Facebook sues the licensee prompting them to take "other action" by publishing the action on a blog or elsewhere. Many expressed concerns that Facebook could unfairly exploit the termination clause or that integrating React into a product might complicate a startup company's future acquisition.<ref>{{cite web|title=A compelling reason not to use ReactJS|first=Austin|last=Liu|url=https://medium.com/bits-and-pixels/a-compelling-reason-not-to-use-reactjs-beac24402f7b|website=Medium}}</ref>

Based on community feedback, Facebook updated the patent grant in April 2015 to be less ambiguous and more permissive:<ref>{{cite web|title=Updating Our Open Source Patent Grant|url=https://code.facebook.com/posts/1639473982937255/updating-our-open-source-patent-grant/}}</ref>

<blockquote>The license granted hereunder will terminate, automatically and without notice, if you (or any of your subsidiaries, corporate affiliates or agents) initiate directly or indirectly, or take a direct financial interest in, any Patent Assertion: (i) against Facebook or any of its subsidiaries or corporate affiliates, (ii) against any party if such Patent Assertion arises in whole or in part from any software, technology, product or service of Facebook or any of its subsidiaries or corporate affiliates, or (iii) against any party relating to the Software. [...] A "Patent Assertion" is any lawsuit or other action alleging direct, indirect, or contributory infringement or inducement to infringe any patent, including a cross-claim or counterclaim.<ref>{{cite web|title=Additional Grant of Patent Rights Version 2|url=https://github.com/facebook/react/blob/b8ba8c83f318b84e42933f6928f231dc0918f864/PATENTS|website=GitHub}}</ref></blockquote>

The [[Apache Software Foundation]] considered this licensing arrangement to be incompatible with its licensing policies, as it "passes along risk to downstream consumers of our software imbalanced in favor of the licensor, not the licensee, thereby violating our Apache legal policy of being a universal donor", and "are not a subset of those found in the [Apache License 2.0], and they cannot be sublicensed as [Apache License 2.0]".<ref>{{Cite web|url=https://www.apache.org/legal/resolved.html|title=ASF Legal Previously Asked Questions|publisher=Apache Software Foundation|language=en|access-date=2017-07-16}}</ref> In August 2017, Facebook dismissed the Apache Foundation's downstream concerns and refused to reconsider their license.<ref>{{Cite web|url=https://code.facebook.com/posts/112130496157735/explaining-react-s-license/|title=Explaining React's License|website=Facebook|access-date=2017-08-18|language=en}}</ref><ref>{{Cite web|url=https://github.com/facebook/react/issues/10191#issuecomment-323486580|title=Consider re-licensing to AL v2.0, as RocksDB has just done|website=Github|language=en|access-date=2017-08-18}}</ref> The following month, [[WordPress]] decided to switch its Gutenberg and Calypso projects away from React.<ref>{{Cite web|url= https://techcrunch.com/2017/09/15/wordpress-to-ditch-react-library-over-facebook-patent-clause-risk/|title= WordPress to ditch React library over Facebook patent clause risk |website=TechCrunch|language=en|access-date=2017-09-16}}</ref>

On September 23, 2017, Facebook announced that the following week, it would re-license Flow, Jest, React, and Immutable.js under a standard [[MIT License]]; the company stated that React was "the foundation of a broad ecosystem of open source software for the web", and that they did not want to "hold back forward progress for nontechnical reasons".<ref>{{Cite web|url= https://code.facebook.com/posts/300798627056246/relicensing-react-jest-flow-and-immutable-js/|title= Relicensing React, Jest, Flow, and Immutable.js |website=Facebook Code|language=en|date=2017-09-23}}</ref>

On September 26, 2017, React 16.0.0 was released with the MIT license.<ref>{{cite web |url=https://reactjs.org/blog/2017/09/26/react-v16.0.html#mit-licensed|title= React v16.0§MIT licensed |last=Clark |first=Andrew |date=September 26, 2017 |website=React Blog}}</ref> The MIT license change has also been backported to the 15.x release line with React 15.6.2.<ref>{{cite web |url=https://reactjs.org/blog/2017/09/25/react-v15.6.2.html |title=React v15.6.2 |last=Hunzaker |first=Nathan |date=September 25, 2017 |website=React Blog}}</ref>

==See also==
{{Portal|Free and open-source software}}
*[[Angular (web framework)]]
*[[Backbone.js]]
*[[Ember.js]]
*[[Gatsby (JavaScript framework)]]
*[[Next.js]]
*[[Svelte]]
*[[Vue.js]]
*[[Comparison of JavaScript-based web frameworks]]
*[[Web Components]]

==References==
{{Reflist|2}}

==Bibliography==
{{Refbegin}}
*{{cite book |last=Larsen |first=John |title=React Hooks in Action With Suspense and Concurrent Mode |year=2021 |publisher=Manning |isbn=978-1720043997}}
*{{cite book |last1=Schwarzmüller |first1=Max |date=2018-05-01 |title=React - The Complete Guide (incl. Hooks, React Router and Redux) |publisher=[[Packt Publishing]] |language=en-US}}
*{{cite book |last=Wieruch |first=Robin |title=The Road to React |publisher=Leanpub |isbn=978-1720043997 |year=2020}}
*{{cite news |last=Dere |first=Mohan |date=2017-12-21 |title=How to integrate create-react-app with all the libraries you need to make a great app |language=en-US |work=freeCodeCamp |url=https://medium.freecodecamp.org/integrating-create-react-app-redux-react-router-redux-observable-bootstrap-altogether-216db97e89a3 |access-date=2018-06-14}}
*{{cite news |last=Panchal |first=Krunal |date=2022-04-26 |title=Angular vs React Detailed Comparison |language=en-US |work=SitePoint |url=https://www.sitepoint.com/angular-vs-react/ |access-date=2023-06-05}}
*{{cite news |last1=Hámori |first1=Fenerec |title=The History of React.js on a Timeline |url=https://blog.risingstack.com/the-history-of-react-js-on-a-timeline/ |access-date=2023-06-05 |work=RisingStack |date=2022-05-31}}
*{{cite news |last=Lardinois |first=Frederic |url=https://techcrunch.com/2017/04/18/facebook-announces-react-fiber-a-rewrite-of-its-react-framework/ |title=Facebook announces React Fiber, a rewrite of its React library |publisher=TechCrunch |date=2017-04-18 |access-date=2023-06-05}}
*{{cite web |title=React Fiber Architecture |url=https://github.com/acdlite/react-fiber-architecture |website=Github |access-date=19 April 2017}}
*{{cite web |last=Baer |first=Eric |title=Chapter 1. What Is React? - What React Is and Why It Matters [Book] |url=https://www.oreilly.com/library/view/what-react-is/9781491996744/ch01.html |access-date=2023-06-05 |website=www.oreilly.com |language=en}}
*{{cite web |last=Krill |first=Paul |date=2014-05-14 |title=React: Making faster, smoother UIs for data-driven Web apps |url=https://www.infoworld.com/article/2608181/javascript/react--making-faster--smoother-uis-for-data-driven-web-apps.html |access-date=2023-06-05 |website=[[InfoWorld]]}}
*{{cite web |title=React - A JavaScript library for building user interfaces. |url=https://reactjs.org |access-date=2023-06-05 |website=reactjs.org |language=en-US}}
{{Refend}}
*{{cite news |last=Jadhav |first=Prashant |date=2023-05-19 |title=ReactJS & NodeJS: An Ideal Web App Development Combination |language=en-US |work=Artoon Solutions |url=https://artoonsolutions.com/combination-of-reactjs-and-nodejs/ |access-date=2023-06-05}}

*{{cite news |last=DC|first=Kumawat|date=2022-01-03 |title=6 Fundamental of React Js App Development will change the way you think! |language=en-US |work=Orio InfoSolutions |url=https://www.orioninfosolutions.com/blog/6-fundamental-of-react-js-app-development-will-change-the-way-you-think |access-date=2022-01-03}}

==External links==
* {{Official website}}
* [https://github.com/facebook/react Github]
* [https://www.facebook.com/react Facebook]
* [https://twitter.com/reactjs Twitter]

{{JS templating |state=autocollapse}}
{{Web frameworks}}
{{ECMAScript}}
{{Facebook navbox}}
{{Authority control}}

[[Category:2015 software]]
[[Category:Ajax (programming)]]
[[Category:Facebook software]]
[[Category:JavaScript libraries]]
[[Category:Software using the MIT license]]
[[Category:Web applications]]
