---
layout: archive
title: "Posts by Year"
permalink: /posts/
author_profile: true
---

{% assign posts_by_year = site.posts | group_by_exp:"post", "post.date | date: '%Y'" %}
{% for year in posts_by_year %}
  <h2 id="{{ year.name | slugify }}" class="archive__subtitle">
    {{ year.name }} <span class="post-count">({{ year.items | size }} posts)</span>
  </h2>
  <ul class="posts">
    {% for post in year.items %}
      <li>
        <h3 class="post-title">
          <a class="post-link" href="{{ post.url }}">{{ post.title }}</a>
        </h3>
        <span class="post-meta">{{ post.date | date: "%B %-d, %Y" }}</span> <!-- 작성일을 제목 아래에 표시 -->
      </li>
    {% endfor %}
  </ul>
{% endfor %}
