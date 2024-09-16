---
layout: archive
title: "All Posts"
permalink: /posts/
author_profile: true
---

## My Posts

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a> - {{ post.date | date: "%b %-d, %Y" }}
    </li>
  {% endfor %}
</ul>
