### 2_1. preparing EURL data ###

# establish connection
temp <- tempfile()
download.file("https://static.data.gouv.fr/resources/horaires-theoriques-du-reseau-europeen-eurolines-isilines-gtfs/20190527-171834/google-transit.zip",temp)

#### reading GTFS files ####

# 1. stop_times

#stop_times <- read.table(here::here("raw-data", "20190527_EURL", "stop_times.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fileEncoding = "UTF-8-BOM")
stop_times <- read.table(unz(temp, "stop_times.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fileEncoding = "UTF-8-BOM")
stop_times[is.na(stop_times)] <- 0
stop_times$sequence_id <- as.numeric(rownames(stop_times))
stop_times <- stop_times[, c("trip_id", "departure_time", "arrival_time", "stop_id", "stop_sequence", "sequence_id", "pickup_type", "drop_off_type")]


# 2. trips

#trips <- read.table(here::here("raw-data", "20190527_EURL", "trips.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fileEncoding = "UTF-8-BOM")
trips <- read.table(unz(temp, "trips.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fill = TRUE, quote = "", fileEncoding = "UTF-8-BOM")
trips <- trips %>% 
  dplyr::select(trip_id, route_id, service_id)

# 3. calendar/calendar dates

#calendar <- read.table(here::here("raw-data", "20190527_EURL", "calendar.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fileEncoding = "UTF-8-BOM")
calendar <- read.table(unz(temp, "calendar.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fill = TRUE, quote = "", fileEncoding = "UTF-8-BOM")
tmp_calendar <- calendar %>% 
  dplyr::select(service_id, start_date, end_date, monday, tuesday, wednesday, thursday, friday, saturday, sunday)

#4. stops

#stops <- read.delim(here::here("raw-data", "20190527_EURL", "stops.txt"), header = TRUE, stringsAsFactors = FALSE, sep = ",", fileEncoding = "UTF-8")
stops <- read.delim(unz(temp, "stops.txt"), header = TRUE, stringsAsFactors = FALSE, sep = ",", fileEncoding = "UTF-8")

stops <- stops %>%
  dplyr::select(-zone_id, -location_type, - parent_station)

#### adresses ####

#### nodes ####
nodes <- stops %>% 
  reverse_geocode_tbl(stop_lon, stop_lat) %>% 
  filter(!is.na(result_label)) %>% #retrieve addresses in France
  glimpse %>%
  separate(result_context, c("dep_id", "department", "region"), sep ="\\,\\s") %>%
  dplyr::rename(stop_lat = latitude, stop_lon = longitude, address = result_label, postcode = result_postcode, city = result_city, insee_code = result_citycode) %>%
  dplyr::select(stop_id, stop_name, stop_lat, stop_lon, insee_code, postcode, region, dep_id, department, city, address)

# assigning values for not recognised adresses

nodes$insee_code[nodes$stop_id == "NB:Stop:94"] <- "79048" 
nodes$postcode[nodes$stop_id == "NB:Stop:94"] <- "79260"
nodes$region[nodes$stop_id == "NB:Stop:94"] <- "Poitou-Charentes"
nodes$dep_id[nodes$stop_id == "NB:Stop:94"] <- "79"
nodes$department[nodes$stop_id == "NB:Stop:94"] <- "Deux-Sèvres"
nodes$city[nodes$stop_id == "NB:Stop:94"] <- "La Crèche"
nodes$address[nodes$stop_id == "NB:Stop:94"] <- "79260 La Crèche"

nodes$insee_code[nodes$stop_id == "NB:Stop:126"] <- "19275" 
nodes$postcode[nodes$stop_id == "NB:Stop:126"] <- "19200"
nodes$region[nodes$stop_id == "NB:Stop:126"] <- "Limousin"
nodes$dep_id[nodes$stop_id == "NB:Stop:126"] <- "19"
nodes$department[nodes$stop_id == "NB:Stop:126"] <- "Corrèze"
nodes$city[nodes$stop_id == "NB:Stop:126"] <- "Ussel"
nodes$address[nodes$stop_id == "	NB:Stop:126"] <- "N89, 19200 Ussel"

nodes$insee_code[nodes$stop_id == "NB:Stop:92"] <- "06088" 
nodes$postcode[nodes$stop_id == "NB:Stop:92"] <- "06200"
nodes$region[nodes$stop_id == "NB:Stop:92"] <- "Provence-Alpes-Côte d'Azur"
nodes$dep_id[nodes$stop_id == "NB:Stop:92"] <- "06"
nodes$department[nodes$stop_id == "NB:Stop:92"] <- "Alpes-Maritimes"
nodes$city[nodes$stop_id == "NB:Stop:92"] <- "Nice"
nodes$address[nodes$stop_id == "NB:Stop:92"] <- "Boulevard Jacqueline Auriol Supérieur, 06200 Nice"

# remove stops for which geoinformation wasn't identified
# relevant if the information provided by banR is changed
nodes <- nodes %>% drop_na("insee_code")

#### edges ####
# processing data frames to convert them into nodes and edges

library("dplyr")
tmp_stop_times <- left_join(x = stop_times, y = trips, by = "trip_id")
tmp_stop_times <- left_join(x = tmp_stop_times, y = calendar, by = "service_id")
tmp_stop_times <- tmp_stop_times %>%
  filter(start_date <= end & end_date >= start) %>%
  arrange(sequence_id)  %>%
  mutate(start_date=pmax(start_date, start), end_date=pmin(end_date, end),
         monday=sum(weekdays(seq(as.Date(start), as.Date(end), "days"))=="Monday")*monday/(week(ymd(end))-week(ymd(start))+1),
         tuesday=sum(weekdays(seq(as.Date(start), as.Date(end), "days"))=="Tuesday")*tuesday/(week(ymd(end))-week(ymd(start))+1),
         wednesday=sum(weekdays(seq(as.Date(start), as.Date(end), "days"))=="Wednesday")*wednesday/(week(ymd(end))-week(ymd(start))+1),
         thursday=sum(weekdays(seq(as.Date(start), as.Date(end), "days"))=="Thursday")*thursday/(week(ymd(end))-week(ymd(start))+1),
         friday=sum(weekdays(seq(as.Date(start), as.Date(end), "days"))=="Friday")*friday/(week(ymd(end))-week(ymd(start))+1),
         saturday=sum(weekdays(seq(as.Date(start), as.Date(end), "days"))=="Saturday")*saturday/(week(ymd(end))-week(ymd(start))+1),
         sunday=sum(weekdays(seq(as.Date(start), as.Date(end), "days"))=="Sunday")*sunday/(week(ymd(end))-week(ymd(start))+1))%>%
  mutate(frequency=monday + tuesday + wednesday + thursday + friday + saturday + sunday)


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
  #filter(arr_sequence_id - dep_sequence_id == 1) %>%
  group_by(departure_placecode, arrival_placecode) %>% 
  dplyr::summarise(frequency = sum(frequency)) %>%
  filter(frequency > 0) %>%
  inner_join(nodes, by = c("departure_placecode" = "stop_id")) %>%
  inner_join(nodes, by = c("arrival_placecode" = "stop_id"), suffix = c("_from", "_to"))

# calculate distance between nodes
edges$distance_vin <- round(geosphere::distVincentyEllipsoid(edges[,c("stop_lat_from","stop_lon_from")], edges[,c("stop_lat_to","stop_lon_to")])/1000,2)
edges$distance_vin <- as.integer(edges$distance_vin)

edges <- edges %>%
  mutate(distance=distance_vin) %>% 
  filter(distance>=100)

edges_EURL_pspace <- edges %>% dplyr::select(departure_placecode, arrival_placecode, frequency, distance)
edges_EURL_pspace$company <- "EURL"

#### L-space edges ####

edges <- edges_detailed %>% 
  filter(arr_sequence_id - dep_sequence_id == 1) %>%
  group_by(departure_placecode, arrival_placecode) %>% 
  dplyr::summarise(frequency = sum(frequency)) %>%
  filter(frequency > 0) %>%
  inner_join(nodes, by = c("departure_placecode" = "stop_id")) %>%
  inner_join(nodes, by = c("arrival_placecode" = "stop_id"), suffix = c("_from", "_to"))

# calculate distance between nodes
edges$distance_vin <- round(geosphere::distVincentyEllipsoid(edges[,c("stop_lat_from","stop_lon_from")], edges[,c("stop_lat_to","stop_lon_to")])/1000,2)
edges$distance_vin <- as.integer(edges$distance_vin)

edges <- edges %>%
  mutate(distance=distance_vin) %>% 
  filter(distance>=100)

edges_EURL_lspace <- edges %>% dplyr::select(departure_placecode, arrival_placecode, frequency, distance)
edges_EURL_lspace$company <- "EURL"
nodes_EURL <- nodes
nodes_EURL$company <- "EURL"

unlink(temp)
