### 2_1. preparing FLIX data ###

# establish connection and downloading files
url <- 'https://openmobilitydata.org/p/flixbus/795/20190420/download'
temp  <- tempfile(fileext = ".zip")
if (file.exists(temp))  'file alredy exists' else download.file(url, temp, mode="wb")
unzip(zipfile = temp,exdir = here::here("raw-data", "FLIX"))

#### reading GTFS files ####

# 1. stop_times

#stop_times <- read.table(here::here("raw-data", "20190420_FLIX", "stop_times.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fileEncoding = "UTF-8-BOM")
stop_times <- read.table(here::here("raw-data", "FLIX", "stop_times.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fileEncoding = "UTF-8-BOM")
stop_times$sequence_id <- as.numeric(rownames(stop_times))
stop_times <- stop_times[, c("trip_id", "departure_time", "arrival_time", "stop_id", "stop_sequence", "sequence_id", "pickup_type", "drop_off_type")]

# 2. trips

#trips <- read.table(here::here("raw-data", "20190420_FLIX", "trips.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fill = TRUE, quote = "")
trips <- read.table(here::here("raw-data", "FLIX", "trips.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fill = TRUE, quote = "")
trips <- trips[, c("trip_id", "route_id", "service_id")]

# 3. calendar/calendar dates

#calendar <- read.table(here::here("raw-data", "20190420_FLIX", "calendar.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fileEncoding = "UTF-8-BOM")
calendar <- read.table(here::here("raw-data", "FLIX", "calendar.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fileEncoding = "UTF-8-BOM")
calendar <- calendar[, c("service_id", "start_date", "end_date", "monday", "tuesday", "wednesday", "thursday", 
                         "friday", "saturday", "sunday")]

calendar_dates <- read.table(here::here("raw-data", "FLIX", "calendar_dates.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fileEncoding = "UTF-8-BOM")

#4. stops
#stops <- read_delim(here::here("raw-data", "20190420_FLIX", "stops.txt"), delim = ",")
stops <- read_delim(here::here("raw-data", "FLIX", "stops.txt"), delim = ",")

#### nodes ####

nodes <- stops %>%
  filter(stop_timezone == "Europe/Paris") %>%
  banR::reverse_geocode_tbl(stop_lon, stop_lat) %>% 
  glimpse %>%
  separate(result_context, c("dep_id", "department", "region"), sep ="\\,\\s") %>%
  dplyr::rename(stop_lat = latitude, stop_lon = longitude, address = result_label, postcode = result_postcode, city = result_city, insee_code = result_citycode) %>%
  dplyr::select(stop_id, stop_name, stop_lat, stop_lon, insee_code, postcode, region, dep_id, department, city, address)

# adding information for stops which can't be parsed

nodes$insee_code[nodes$stop_id == "FLIXBUS:19508"] <- "29026" 
nodes$postcode[nodes$stop_id == "FLIXBUS:19508"] <- "29150"
nodes$region[nodes$stop_id == "FLIXBUS:19508"] <- "Bretagne"
nodes$dep_id[nodes$stop_id == "FLIXBUS:19508"] <- "29"
nodes$department[nodes$stop_id == "FLIXBUS:19508"] <- "Finistère"
nodes$city[nodes$stop_id == "FLIXBUS:19508"] <- "Châteaulin"
nodes$address[nodes$stop_id == "FLIXBUS:19508"] <- "29150 Châteaulin"

nodes$insee_code[nodes$stop_id == "FLIXBUS:9658"] <- "62318" 
nodes$postcode[nodes$stop_id == "FLIXBUS:9658"] <- "62630"
nodes$region[nodes$stop_id == "FLIXBUS:9658"] <- "Nord-Pas-de-Calais"
nodes$dep_id[nodes$stop_id == "FLIXBUS:9658"] <- "62"
nodes$department[nodes$stop_id == "FLIXBUS:9658"] <- "Pas-de-Calais"
nodes$city[nodes$stop_id == "FLIXBUS:9658"] <- "Étaples"
nodes$address[nodes$stop_id == "FLIXBUS:9658"] <- "62630 Étaples"

nodes$insee_code[nodes$stop_id == "FLIXBUS:7778"] <- "06088" 
nodes$postcode[nodes$stop_id == "FLIXBUS:7778"] <- "06200"
nodes$region[nodes$stop_id == "FLIXBUS:7778"] <- "Provence-Alpes-Côte d'Azur"
nodes$dep_id[nodes$stop_id == "FLIXBUS:7778"] <- "06"
nodes$department[nodes$stop_id == "FLIXBUS:7778"] <- "Alpes-Maritimes"
nodes$city[nodes$stop_id == "FLIXBUS:7778"] <- "Nice"
nodes$address[nodes$stop_id == "FLIXBUS:7778"] <- "Boulevard Jacqueline Auriol Supérieur, 06200 Nice"

nodes$insee_code[nodes$stop_id == "FLIXBUS:8008"] <- "77132" 
nodes$postcode[nodes$stop_id == "FLIXBUS:8008"] <- "77700"
nodes$region[nodes$stop_id == "FLIXBUS:8008"] <- "Coupvray"
nodes$dep_id[nodes$stop_id == "FLIXBUS:8008"] <- "77"
nodes$department[nodes$stop_id == "FLIXBUS:8008"] <- "Seine-et-Marne"
nodes$city[nodes$stop_id == "FLIXBUS:8008"] <- "Île-de-France"
nodes$address[nodes$stop_id == "FLIXBUS:8008"] <- "77700 Coupvray"

nodes$insee_code[nodes$stop_id == "FLIXBUS:14498"] <- "77132" 
nodes$postcode[nodes$stop_id == "FLIXBUS:14498"] <- "77700"
nodes$region[nodes$stop_id == "FLIXBUS:14498"] <- "Coupvray"
nodes$dep_id[nodes$stop_id == "FLIXBUS:14498"] <- "77"
nodes$department[nodes$stop_id == "FLIXBUS:14498"] <- "Seine-et-Marne"
nodes$city[nodes$stop_id == "FLIXBUS:14498"] <- "Île-de-France"
nodes$address[nodes$stop_id == "FLIXBUS:14498"] <- "77700 Coupvray"

nodes$insee_code[nodes$stop_id == "FLIXBUS:14498"] <- "01200" 
nodes$postcode[nodes$stop_id == "FLIXBUS:14498"] <- "01091"
nodes$region[nodes$stop_id == "FLIXBUS:14498"] <- "	Rhône-Alpes"
nodes$dep_id[nodes$stop_id == "FLIXBUS:14498"] <- "01"
nodes$department[nodes$stop_id == "FLIXBUS:14498"] <- "Ain"
nodes$city[nodes$stop_id == "FLIXBUS:14498"] <- "Châtillon-en-Michaille"
nodes$address[nodes$stop_id == "FLIXBUS:14498"] <- "01091 Châtillon-en-Michaille"

# remove stops for which geoinformation wasn't identified
# relevant if the information provided by banR is changed
nodes <- nodes %>% drop_na("insee_code")

#### edges ####
# processing data frames to extract edges

library("lubridate")

tmp_calendar_dates <- calendar_dates %>%
  filter(date >= start & date <= end) %>%
  mutate(monday = ifelse(weekdays(as.Date(ymd(date))) == "Monday", 1, 0),
         tuesday = ifelse(weekdays(as.Date(ymd(date))) == "Tuesday", 1, 0),
         wednesday = ifelse(weekdays(as.Date(ymd(date))) == "Wednesday", 1, 0),
         thursday = ifelse(weekdays(as.Date(ymd(date))) == "Thursday", 1, 0),
         friday = ifelse(weekdays(as.Date(ymd(date))) == "Friday", 1, 0),
         saturday = ifelse(weekdays(as.Date(ymd(date))) == "Saturday", 1, 0),
         sunday = ifelse(weekdays(as.Date(ymd(date))) == "Sunday", 1, 0)) 

tmp_added_services <- subset(tmp_calendar_dates, tmp_calendar_dates$exception_type == 1) %>%
  dplyr::select(-exception_type, -date) %>%
  group_by(service_id) %>%
  summarise_all(funs(sum))

tmp_deleted_services <- subset(tmp_calendar_dates, tmp_calendar_dates$exception_type == 2) %>%
  dplyr::select(-exception_type, -date) %>%
  group_by(service_id) %>%
  summarise_all(funs(sum))

tmp_calendar <- calendar %>%
  filter(start_date <= end & end_date >= start) %>%
  full_join(tmp_added_services, by = "service_id",  suffix = c("", ".add")) %>%
  full_join(tmp_deleted_services, by = "service_id",  suffix = c("", ".del"))  %>%
  replace(is.na(.), 0) %>%
  filter(start_date <= end & end_date >= start) %>%
  mutate(start_date=pmax(start_date, start), end_date=pmin(end_date, end)) %>% 
  mutate(monday=(sum(weekdays(seq(as.Date(start), as.Date(end), "days"))=="Monday")*monday + monday.add - monday.del)/(week(ymd(end))-week(ymd(start))+1),
         tuesday=(sum(weekdays(seq(as.Date(start), as.Date(end), "days"))=="Tuesday")*tuesday + tuesday.add - tuesday.del)/(week(ymd(end))-week(ymd(start))+1),
         wednesday=(sum(weekdays(seq(as.Date(start), as.Date(end), "days"))=="Wednesday")*wednesday + wednesday.add - wednesday.del)/(week(ymd(end))-week(ymd(start))+1),
         thursday=(sum(weekdays(seq(as.Date(start), as.Date(end), "days"))=="Thursday")*thursday + thursday.add - thursday.del)/(week(ymd(end))-week(ymd(start))+1),
         friday=(sum(weekdays(seq(as.Date(start), as.Date(end), "days"))=="Friday")*friday + friday.add - friday.del)/(week(ymd(end))-week(ymd(start))+1),
         saturday=(sum(weekdays(seq(as.Date(start), as.Date(end), "days"))=="Saturday")*saturday + saturday.add - saturday.del)/(week(ymd(end))-week(ymd(start))+1),
         sunday=(sum(weekdays(seq(as.Date(start), as.Date(end), "days"))=="Sunday")*sunday + sunday.add - sunday.del)/(week(ymd(end))-week(ymd(start))+1)) %>%
  mutate(frequency=monday + tuesday + wednesday + thursday + friday + saturday + sunday) %>% 
  dplyr::select(service_id, start_date, end_date, frequency) 


tmp_stop_times <- left_join(x = stop_times, y = trips, by = "trip_id")
tmp_stop_times <- left_join(x = tmp_stop_times, y = tmp_calendar, by = "service_id")
tmp_stop_times <- tmp_stop_times %>%  arrange(sequence_id)

# extractig station pairs

edges_detailed <- data.table::setDT(tmp_stop_times)[, data.table::transpose(combn(sequence_id, 2, FUN = list)), by = trip_id]
data.table::setnames(edges_detailed, old=c("V1", "V2"), new=c("dep_sequence_id", "arr_sequence_id"))

#### P-space edges ####

edges_detailed <- edges_detailed %>%
  inner_join(tmp_stop_times[, c("route_id", "trip_id", "service_id", "stop_id", "sequence_id", "pickup_type", "departure_time", "frequency")], by = c("dep_sequence_id" = "sequence_id"), suffix = c("",".dep")) %>%
  dplyr::select(!trip_id.dep) %>%
  inner_join(tmp_stop_times[, c("stop_id", "sequence_id", "drop_off_type", "arrival_time")], by = c("arr_sequence_id" = "sequence_id"), suffix = c(".dep",".arr")) %>%
  filter(pickup_type == 0 & drop_off_type == 0) %>% 
  dplyr::rename(departure_placecode = stop_id.dep, arrival_placecode = stop_id.arr) %>% 
  dplyr::select(route_id, trip_id, service_id, departure_placecode, dep_sequence_id, arrival_placecode, arr_sequence_id, departure_time, arrival_time, frequency)

edges <- edges_detailed %>%
  #filter(arr_sequence_id - dep_sequence_id == 1) %>% disabled for p-space
  group_by(departure_placecode, arrival_placecode) %>% 
  dplyr::summarise(frequency = sum(frequency)) %>%
  filter(frequency > 0) %>%
  inner_join(nodes, by = c("departure_placecode" = "stop_id")) %>%
  inner_join(nodes, by = c("arrival_placecode" = "stop_id"), suffix = c("_from", "_to"))

edges$distance_vin <- round(geosphere::distVincentyEllipsoid(edges[,c("stop_lat_from","stop_lon_from")], edges[,c("stop_lat_to","stop_lon_to")])/1000,2)
edges$distance_vin <- as.integer(edges$distance_vin)

edges <- edges %>%
  mutate(distance=distance_vin) %>% 
  filter(distance>=100)

edges_FLIX_pspace <- edges %>% dplyr::select(departure_placecode, arrival_placecode, frequency, distance)
edges_FLIX_pspace$company <- "FLIX"

#### L-space edges ####

edges <- edges_detailed %>% 
  filter(arr_sequence_id - dep_sequence_id == 1) %>%
  group_by(departure_placecode, arrival_placecode) %>% 
  dplyr::summarise(frequency = sum(frequency)) %>%
  filter(frequency > 0) %>%
  inner_join(nodes, by = c("departure_placecode" = "stop_id")) %>%
  inner_join(nodes, by = c("arrival_placecode" = "stop_id"), suffix = c("_from", "_to"))

edges$distance_vin <- round(geosphere::distVincentyEllipsoid(edges[,c("stop_lat_from","stop_lon_from")], edges[,c("stop_lat_to","stop_lon_to")])/1000,2)
edges$distance_vin <- as.integer(edges$distance_vin)


edges <- edges %>%
  mutate(distance=distance_vin) %>% 
  filter(distance>=100)

edges_FLIX_lspace <- edges %>% dplyr::select(departure_placecode, arrival_placecode, frequency, distance)
edges_FLIX_lspace$company <- "FLIX"

nodes_FLIX <- nodes
nodes_FLIX$company <- "FLIX"

unlink(temp)
unlink(here::here("raw-data", "FLIX"), recursive = T)

