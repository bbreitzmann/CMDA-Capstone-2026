library(readr)
library(tidyverse)
library(reticulate)


# File paths
meta_path <- "Documents/VT_Undergrad/Spring_25:26/CMDA_4864_CAPSTONE/team_repo/meta.csv"
ctx_path  <- "Documents/VT_Undergrad/Spring_25:26/CMDA_4864_CAPSTONE/team_repo/X_ctx.csv"
x_ts_path <- "Documents/VT_Undergrad/Spring_25:26/CMDA_4864_CAPSTONE/team_repo/X_ts.npy"


# Load data
meta <- read_csv(meta_path, show_col_types = FALSE)
ctx  <- read_csv(ctx_path, show_col_types = FALSE)
np <- import("numpy")
X_ts <- np$load(x_ts_path)

# convert python array to R array
X_ts <- py_to_r(X_ts)

dim(X_ts) # we will assign channel headers later on

# Basic checks
glimpse(meta)
glimpse(ctx)

cat("Rows in meta:", nrow(meta), "\n")
cat("Rows in ctx :", nrow(ctx), "\n")
cat("X_ts dimensions:", paste(dim(X_ts), collapse = " x "), "\n")


# human-readable labels
label_levels <- c("Conflict", "Bump", "Hard Brake", "Not an SCE")

meta <- meta %>%
  mutate(
    y = as.integer(y),
    event_type_name = factor(
      case_when(
        y == 0 ~ "Conflict",
        y == 1 ~ "Bump",
        y == 2 ~ "Hard Brake",
        y == 3 ~ "Not an SCE",
        TRUE   ~ "Unknown"
      ),
      levels = label_levels
    ),
    event_idx = row_number()
  )

ctx <- ctx %>%
  mutate(
    y = as.integer(y),
    EVENT_TYPE = as.integer(EVENT_TYPE),
    event_type_name = factor(
      case_when(
        y == 0 ~ "Conflict",
        y == 1 ~ "Bump",
        y == 2 ~ "Hard Brake",
        y == 3 ~ "Not an SCE",
        TRUE   ~ "Unknown"
      ),
      levels = label_levels
    ),
    weather = replace_na(weather, "undefined"),
    scene = replace_na(scene, "undefined"),
    timeofday = replace_na(timeofday, "undefined")
  )


# Merge static context w/ meta
eda_df <- ctx %>%
  left_join(
    meta %>% select(BDD_ID, EVENT_ID, event_time_ms, anchor_source, label_ts_ms),
    by = c("BDD_ID", "EVENT_ID")
  )

glimpse(eda_df)


# Dataset overview
dataset_overview <- tibble(
  metric = c(
    "usable_events",
    "unique_bdd_ids",
    "unique_event_ids",
    "ts_num_events",
    "ts_num_channels",
    "ts_num_timesteps"
  ),
  value = c(
    nrow(meta),
    n_distinct(meta$BDD_ID),
    n_distinct(meta$EVENT_ID),
    dim(X_ts)[1],
    dim(X_ts)[2],
    dim(X_ts)[3]
  )
)

print(dataset_overview)

events_per_bdd <- meta %>%
  count(BDD_ID, name = "n_events")

summary(events_per_bdd$n_events)

ggplot(events_per_bdd, aes(x = n_events)) +
  geom_histogram(binwidth = 1) +
  labs(
    title = "Events per BDD_ID",
    x = "Number of Events",
    y = "Count"
  )

meta %>%
  count(anchor_source) %>%
  ggplot(aes(x = anchor_source, y = n)) +
  geom_col() +
  labs(
    title = "Anchor Source Used for Event Window Construction",
    x = "Anchor Source",
    y = "Count"
  )


# Label distribution and class balance
label_counts <- meta %>%
  count(event_type_name) %>%
  mutate(prop = n / sum(n))

print(label_counts)

ggplot(label_counts, aes(x = event_type_name, y = n)) +
  geom_col() +
  labs(
    title = "EVENT_TYPE Distribution",
    x = "Event Type",
    y = "Count"
  )

ggplot(label_counts, aes(x = event_type_name, y = prop)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "EVENT_TYPE Proportion",
    x = "Event Type",
    y = "Proportion"
  )


# 3. STATIC SCENE / CONTEXT EDA
ctx %>%
  count(weather) %>%
  ggplot(aes(x = reorder(weather, -n), y = n)) +
  geom_col() +
  labs(
    title = "Weather Distribution",
    x = "Weather",
    y = "Count"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ctx %>%
  count(weather, event_type_name) %>%
  ggplot(aes(x = weather, y = n, fill = event_type_name)) +
  geom_col(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Event Type Composition by Weather",
    x = "Weather",
    y = "Proportion",
    fill = "Event Type"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ctx %>%
  count(weather, event_type_name) %>%
  ggplot(aes(x = weather, y = n, fill = event_type_name)) +
  geom_col(position = "dodge") +
  labs(
    title = "Event Type Counts by Weather",
    x = "Weather",
    y = "Count",
    fill = "Event Type"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ctx %>%
  count(timeofday) %>%
  ggplot(aes(x = reorder(timeofday, -n), y = n)) +
  geom_col() +
  labs(
    title = "Time of Day Distribution",
    x = "Time of Day",
    y = "Count"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ctx %>%
  count(timeofday, event_type_name) %>%
  ggplot(aes(x = timeofday, y = n, fill = event_type_name)) +
  geom_col(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Event Type Composition by Time of Day",
    x = "Time of Day",
    y = "Proportion",
    fill = "Event Type"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ctx %>%
  count(timeofday, event_type_name) %>%
  ggplot(aes(x = timeofday, y = n, fill = event_type_name)) +
  geom_col(position = "dodge") +
  labs(
    title = "Event Type Counts by Time of Day",
    x = "Time of Day",
    y = "Count",
    fill = "Event Type"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ctx %>%
  count(scene) %>%
  ggplot(aes(x = reorder(scene, -n), y = n)) +
  geom_col() +
  labs(
    title = "Scene Distribution",
    x = "Scene",
    y = "Count"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ctx %>%
  count(scene, event_type_name) %>%
  ggplot(aes(x = scene, y = n, fill = event_type_name)) +
  geom_col(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title = "Event Type Composition by Scene",
    x = "Scene",
    y = "Proportion",
    fill = "Event Type"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Object environment eda
object_features <- c(
  "n_car",
  "n_pedestrian",
  "n_truck",
  "n_bus",
  "n_traffic_light",
  "n_traffic_sign",
  "total_boxed_objects"
)

object_summary <- ctx %>%
  group_by(event_type_name) %>%
  summarise(
    across(
      all_of(object_features),
      list(mean = ~mean(.x, na.rm = TRUE),
           median = ~median(.x, na.rm = TRUE))
    ),
    .groups = "drop"
  )

print(object_summary)

ctx %>%
  select(event_type_name, all_of(object_features)) %>%
  pivot_longer(
    cols = -event_type_name,
    names_to = "feature",
    values_to = "value"
  ) %>%
  ggplot(aes(x = event_type_name, y = value)) +
  geom_boxplot() +
  facet_wrap(~feature, scales = "free_y") +
  labs(
    title = "Object Environment Features by Event Type",
    x = "Event Type",
    y = "Value"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ctx %>%
  group_by(event_type_name) %>%
  summarise(across(all_of(object_features), ~mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  pivot_longer(
    cols = -event_type_name,
    names_to = "feature",
    values_to = "mean_value"
  ) %>%
  ggplot(aes(x = feature, y = mean_value, fill = event_type_name)) +
  geom_col(position = "dodge") +
  labs(
    title = "Mean Object Features by Event Type",
    x = "Feature",
    y = "Mean Value",
    fill = "Event Type"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Traffic lights
light_features <- c("n_tl_red", "n_tl_yellow", "n_tl_green")

ctx %>%
  group_by(event_type_name) %>%
  summarise(across(all_of(light_features), ~mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  pivot_longer(
    cols = -event_type_name,
    names_to = "feature",
    values_to = "mean_value"
  ) %>%
  ggplot(aes(x = feature, y = mean_value, fill = event_type_name)) +
  geom_col(position = "dodge") +
  labs(
    title = "Average Traffic-Light Features by Event Type",
    x = "Feature",
    y = "Mean Count",
    fill = "Event Type"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Bounding box geometry
geometry_features <- c("min_box_area", "max_box_area", "mean_box_area")

ctx %>%
  select(event_type_name, all_of(geometry_features)) %>%
  pivot_longer(
    cols = -event_type_name,
    names_to = "feature",
    values_to = "value"
  ) %>%
  ggplot(aes(x = event_type_name, y = value)) +
  geom_boxplot() +
  facet_wrap(~feature, scales = "free_y") +
  labs(
    title = "Bounding Box Geometry by Event Type",
    x = "Event Type",
    y = "Area"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ctx %>%
  select(event_type_name, all_of(geometry_features)) %>%
  pivot_longer(
    cols = -event_type_name,
    names_to = "feature",
    values_to = "value"
  ) %>%
  filter(!is.na(value), value > 0) %>%
  ggplot(aes(x = event_type_name, y = value)) +
  geom_boxplot() +
  facet_wrap(~feature, scales = "free_y") +
  scale_y_log10() +
  labs(
    title = "Bounding Box Geometry by Event Type (Log Scale)",
    x = "Event Type",
    y = "Area (log10 scale)"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Occlusion and truncation
quality_features <- c("pct_occluded", "pct_truncated")

ctx %>%
  select(event_type_name, all_of(quality_features)) %>%
  pivot_longer(
    cols = -event_type_name,
    names_to = "feature",
    values_to = "value"
  ) %>%
  ggplot(aes(x = event_type_name, y = value)) +
  geom_boxplot() +
  facet_wrap(~feature, scales = "free_y") +
  labs(
    title = "Occlusion / Truncation by Event Type",
    x = "Event Type",
    y = "Proportion"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Static feature correlation
numeric_ctx <- ctx %>%
  select(
    n_car, n_pedestrian, n_truck, n_bus,
    n_traffic_light, n_traffic_sign,
    n_tl_red, n_tl_yellow, n_tl_green,
    total_boxed_objects,
    min_box_area, max_box_area, mean_box_area,
    pct_occluded, pct_truncated
  )

corr_long <- cor(numeric_ctx, use = "pairwise.complete.obs") %>%
  as.data.frame() %>%
  rownames_to_column("feature_1") %>%
  pivot_longer(
    cols = -feature_1,
    names_to = "feature_2",
    values_to = "correlation"
  )

ggplot(corr_long, aes(x = feature_1, y = feature_2, fill = correlation)) +
  geom_tile() +
  labs(
    title = "Correlation Heatmap of Static Context Features",
    x = NULL,
    y = NULL
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Static summary table
static_summary <- ctx %>%
  group_by(event_type_name) %>%
  summarise(
    n = n(),
    mean_n_car = mean(n_car, na.rm = TRUE),
    mean_n_pedestrian = mean(n_pedestrian, na.rm = TRUE),
    mean_total_boxed_objects = mean(total_boxed_objects, na.rm = TRUE),
    mean_max_box_area = mean(max_box_area, na.rm = TRUE),
    mean_pct_occluded = mean(pct_occluded, na.rm = TRUE),
    .groups = "drop"
  )

print(static_summary)


# Time-series data from X_ts.npy
# Assumes X_ts shape is (N, C, T)
# and row order matches meta row order

channel_names <- c(
  "accel_x", "accel_y", "accel_z",
  "gyro_x", "gyro_y", "gyro_z",
  "speed"
)

stopifnot(dim(X_ts)[1] == nrow(meta))

if (dim(X_ts)[2] != length(channel_names)) {
  stop("Number of channels in X_ts does not match length of channel_names.")
}

# Convert array to long tidy format
ts_long <- as.data.frame.table(X_ts, responseName = "value") %>%
  as_tibble() %>%
  rename(
    event_idx = Var1,
    channel_idx = Var2,
    t_idx = Var3
  ) %>%
  mutate(
    event_idx = as.integer(event_idx),
    channel_idx = as.integer(channel_idx),
    t_idx = as.integer(t_idx),
    channel = channel_names[channel_idx]
  ) %>%
  left_join(
    meta %>%
      select(event_idx, BDD_ID, EVENT_ID, y, event_type_name),
    by = "event_idx"
  )

glimpse(ts_long)


# Mean trajectory by class and channel
ts_mean <- ts_long %>%
  group_by(event_type_name, channel, t_idx) %>%
  summarise(
    mean_value = mean(value, na.rm = TRUE),
    sd_value = sd(value, na.rm = TRUE),
    n = n(),
    se = sd_value / sqrt(n),
    lower = mean_value - se,
    upper = mean_value + se,
    .groups = "drop"
  )

ggplot(ts_mean, aes(x = t_idx, y = mean_value, color = event_type_name)) +
  geom_line() +
  facet_wrap(~channel, scales = "free_y") +
  labs(
    title = "Mean Time-Series Profile by Event Type",
    x = "Resampled Time Index",
    y = "Mean Value",
    color = "Event Type"
  )


# Mean +/- SE ribbon
ggplot(ts_mean, aes(x = t_idx, y = mean_value, color = event_type_name, fill = event_type_name)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
  geom_line() +
  facet_wrap(~channel, scales = "free_y") +
  labs(
    title = "Mean ± SE Time-Series Profile by Event Type",
    x = "Resampled Time Index",
    y = "Value",
    color = "Event Type",
    fill = "Event Type"
  )


# Median trajectory by class and channel
ts_median <- ts_long %>%
  group_by(event_type_name, channel, t_idx) %>%
  summarise(
    median_value = median(value, na.rm = TRUE),
    .groups = "drop"
  )

ggplot(ts_median, aes(x = t_idx, y = median_value, color = event_type_name)) +
  geom_line() +
  facet_wrap(~channel, scales = "free_y") +
  labs(
    title = "Median Time-Series Profile by Event Type",
    x = "Resampled Time Index",
    y = "Median Value",
    color = "Event Type"
  )


# Random example traces per class/channel
set.seed(1)

sampled_events <- ts_long %>%
  distinct(EVENT_ID, event_type_name) %>%
  group_by(event_type_name) %>%
  slice_sample(n = min(5, n())) %>%
  ungroup()

ts_examples <- ts_long %>%
  semi_join(sampled_events, by = c("EVENT_ID", "event_type_name"))

ggplot(ts_examples, aes(x = t_idx, y = value, group = EVENT_ID, color = event_type_name)) +
  geom_line(alpha = 0.4) +
  facet_wrap(~channel, scales = "free_y") +
  labs(
    title = "Random Example Event Windows by Channel",
    x = "Resampled Time Index",
    y = "Value",
    color = "Event Type"
  )


# Focused speed profile plot
ts_mean %>%
  filter(channel == "speed") %>%
  ggplot(aes(x = t_idx, y = mean_value, color = event_type_name)) +
  geom_line(size = 1) +
  labs(
    title = "Average Speed Profile Around Event",
    x = "Resampled Time Index",
    y = "Speed",
    color = "Event Type"
  )


# Focused acceleration plot
ts_mean %>%
  filter(channel == "accel_z") %>%
  ggplot(aes(x = t_idx, y = mean_value, color = event_type_name)) +
  geom_line(size = 1) +
  labs(
    title = "Average accel_z Profile Around Event",
    x = "Resampled Time Index",
    y = "accel_z",
    color = "Event Type"
  )


# Focused gyro plot
ts_mean %>%
  filter(channel == "gyro_y") %>%
  ggplot(aes(x = t_idx, y = mean_value, color = event_type_name)) +
  geom_line(size = 1) +
  labs(
    title = "Average gyro_y Profile Around Event",
    x = "Resampled Time Index",
    y = "gyro_y",
    color = "Event Type"
  )

