# UConn Eats

UConn Eats is a public dining decision app for UConn Storrs that recommends where to eat based on menu match, dietary constraints, distance, and predicted crowd level.

## Project Goals
- Recommend the best open dining hall for the current meal window.
- Suggest the next best time/day if a desired item is not available now.
- Let users create alerts for future menu availability.
- Show occupancy forecasts using 45-minute time blocks.

## Key Features (MVP)
- Public access (no SSO required)
- Web-scraped menu and dining hall data from official UConn Dining pages
- Hard-filtered allergen/dietary constraint handling
- Explainable recommendation results (why this hall)
- Occupancy forecast labels (`Low`, `Medium`, `High`) plus score
- "Not available now" fallback and alert subscriptions

## Data Sources
- Official UConn Dining public website (menus, hall details, hours)
- Internal SQL occupancy data source (45-minute granularity)

## High-Level Architecture
- Menu Ingestion Service: scrape -> parse -> normalize -> store
- Recommendation API: filter + rank halls by user context
- Forecast Service: occupancy baseline and later ML enhancements
- Notification Service: alerts when requested items appear

## Safety and Constraints
- Allergen/dietary restrictions are treated as hard constraints.
- If data cannot verify a requested hard constraint, options are excluded by default.
- The app surfaces official substitution/allergen disclaimers and advises on-site verification for severe allergies.

## Repository Files
- `product-spec.md`: full Product Spec v2
- `README.md`: repository overview and usage context

## Planned Phases
1. Phase 0: scraper + normalized storage + initial UI
2. Phase 1 (MVP): full halls, open-now logic, baseline occupancy, alerts
3. Phase 2: improved NLP matching, stronger forecasting, personalization

## Status
Specification phase complete. Implementation scaffolding next.
