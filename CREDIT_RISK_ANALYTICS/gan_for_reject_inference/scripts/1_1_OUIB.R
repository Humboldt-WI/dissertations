### 2_1. preparing OUIB data ###

# establish connection and downloading files
url <- 'https://transitfeeds.com/p/idbus/519/20190408/download'
temp  <- tempfile(fileext = ".zip")
if (file.exists(temp))  'file alredy exists' else download.file(url, temp, mode="wb")
unzip(zipfile = temp,exdir = here::here("raw-data", "OUIB"))

#### reading GTFS files ####

#1. stop_times

#stop_times <- read.table(here::here("raw-data", "20190408_OUIB", "stop_times.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fileEncoding = "UTF-8-BOM")
stop_times <- read.table(here::here("raw-data", "OUIB", "stop_times.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fileEncoding = "UTF-8-BOM")
stop_times$sequence_id <- as.numeric(rownames(stop_times))
stop_times <- stop_times[, c("trip_id", "departure_time", "arrival_time", "stop_id", "stop_sequence", "sequence_id", "pickup_type", "drop_off_type")]

#2. trips

#trips <- read.table(here::here("raw-data", "20190408_OUIB", "trips.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fill = TRUE, quote = "", fileEncoding = "UTF-8-BOM")
trips <- read.table(here::here("raw-data", "OUIB", "trips.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fill = TRUE, quote = "", fileEncoding = "UTF-8-BOM")
trips <- trips %>% 
  dplyr::select(trip_id, route_id, service_id)

#3. calendar/calendar dates 

#calendar <- read.table(here::here("raw-data", "20190408_OUIB", "calendar.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fileEncoding = "UTF-8-BOM")
calendar <- read.table(here::here("raw-data", "OUIB", "calendar.txt"), stringsAsFactors = FALSE, sep = ",", header = TRUE, fileEncoding = "UTF-8-BOM")
calendar <- calendar %>% 
  dplyr::select(service_id, start_date, end_date, monday, tuesday, wednesday, thursday, friday, saturday, sunday)

#4. stops

#stops <- read.delim(here::here("raw-data", "20190408_OUIB", "stops.txt"), header = TRUE, stringsAsFactors = FALSE, sep = ",", fileEncoding = "UTF-8")
stops <- read.delim(here::here("raw-data", "OUIB", "stops.txt"), header = TRUE, stringsAsFactors = FALSE, sep = ",", fileEncoding = "UTF-8")

#### nodes ####

nodes <- stops %>%
  filter(stop_timezone == "Europe/Paris") %>%
  banR::reverse_geocode_tbl(stop_lon, stop_lat) %>% 
  glimpse %>%
  separate(result_context, c("dep_id", "department", "region"), sep ="\\,\\s") %>%
  dplyr::rename(stop_lat = latitude, stop_lon = longitude, address = result_label, postcode = result_postcode, city = result_city, insee_code = result_citycode) %>%
  dplyr::select(stop_id, stop_name, stop_lat, stop_lon, insee_code, postcode, region, dep_id, department, city, address) %>%
  filter(stop_id != "COD" & stop_id != "SAR")

# adding information for stops which can't be parsed

nodes$insee_code[nodes$stop_id == "ORO"] <- "94054" 
nodes$postcode[nodes$stop_id == "ORO"] <- "94390"
nodes$region[nodes$stop_id == "ORO"] <- "Île-de-France"
nodes$dep_id[nodes$stop_id == "ORO"] <- "75"
nodes$department[nodes$stop_id == "ORO"] <- "Val-de-Marne"
nodes$city[nodes$stop_id == "ORO"] <- "Orly"
nodes$address[nodes$stop_id == "ORO"] <- "4-1 Avenue O, 94390 Paray-Vieille-Poste"

nodes$insee_code[nodes$stop_id == "ABL"] <- "57422" 
nodes$postcode[nodes$stop_id == "ABL"] <- "57420"
nodes$region[nodes$stop_id == "ABL"] <- "Grand Est"
nodes$dep_id[nodes$stop_id == "ABL"] <- "57"
nodes$department[nodes$stop_id == "ABL"] <- "Moselle"
nodes$city[nodes$stop_id == "ABL"] <- "Louvigny"
nodes$address[nodes$stop_id == "ABL"] <- "57420 Louvigny"

nodes$insee_code[nodes$stop_id == "MRP"] <- "13055" 
nodes$postcode[nodes$stop_id == "MRP"] <- "13015"
nodes$region[nodes$stop_id == "MRP"] <- "Provence-Alpes-Côte d'Azur"
nodes$dep_id[nodes$stop_id == "MRP"] <- "13"
nodes$department[nodes$stop_id == "MRP"] <- "Bouches-du-Rhône"
nodes$city[nodes$stop_id == "MRP"] <- "Marignane"
nodes$address[nodes$stop_id == "MRP"] <- "Port de Marseille Fos - Porte 4, Chemin du Littoral, 13015 Marseille"

nodes$insee_code[nodes$stop_id == "XMN"] <- "73257" 
nodes$postcode[nodes$stop_id == "XMN"] <- "73440"
nodes$region[nodes$stop_id == "XMN"] <- "	Rhône-Alpes"
nodes$dep_id[nodes$stop_id == "XMN"] <- "73"
nodes$department[nodes$stop_id == "XMN"] <- "Savoie"
nodes$city[nodes$stop_id == "XMN"] <- "Saint-Martin-de-Belleville"
nodes$address[nodes$stop_id == "XMN"] <- "73440 Saint-Martin-de-Belleville"

nodes$insee_code[nodes$stop_id == "TVC"] <- "73296" 
nodes$postcode[nodes$stop_id == "TVC"] <- "73320"
nodes$region[nodes$stop_id == "TVC"] <- "	Rhône-Alpes"
nodes$dep_id[nodes$stop_id == "TVC"] <- "73"
nodes$department[nodes$stop_id == "TVC"] <- "Savoie"
nodes$city[nodes$stop_id == "TVC"] <- "Tignes"
nodes$address[nodes$stop_id == "TVC"] <- "73320 Tignes"

nodes$insee_code[nodes$stop_id == "XVT"] <- "73257" 
nodes$postcode[nodes$stop_id == "XVT"] <- "73440"
nodes$region[nodes$stop_id == "XVT"] <- "	Rhône-Alpes"
nodes$dep_id[nodes$stop_id == "XVT"] <- "73"
nodes$department[nodes$stop_id == "XVT"] <- "Savoie"
nodes$city[nodes$stop_id == "XVT"] <- "Saint-Martin-de-Belleville"
nodes$address[nodes$stop_id == "XVT"] <- "73440 Saint-Martin-de-Belleville"

# remove stops for which geoinformation wasn't identified
# relevant if the information provided by banR is changed
nodes <- nodes %>% drop_na("insee_code")

#### edges ####
# processing data frames to convert them into nodes and edges

tmp_stop_times <- left_join(x = stop_times, y = trips, by = "trip_id")
tmp_stop_times <- left_join(x = tmp_stop_times, y = calendar, by = "service_id")

tmp_stop_times <- tmp_stop_times %>%
  filter(start_date <= end & end_date >= start) %>%
  arrange(sequence_id) %>%
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

edges$distance_vin <- round(geosphere::distVincentyEllipsoid(edges[,c("stop_lat_from","stop_lon_from")], edges[,c("stop_lat_to","stop_lon_to")])/1000,2)
edges$distance_vin <- as.integer(edges$distance_vin)

edges <- edges %>%
  mutate(distance=distance_vin) %>% 
  filter(distance>=100)

edges_OUIB_pspace <- edges %>% dplyr::select(departure_placecode, arrival_placecode, frequency, distance)
edges_OUIB_pspace$company <- "OUIB"

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

edges_OUIB_lspace <- edges %>% dplyr::select(departure_placecode, arrival_placecode, frequency, distance)
edges_OUIB_lspace$company <- "OUIB"
nodes_OUIB <- nodes
nodes_OUIB$company <- "OUIB"

unlink(temp)
unlink(here::here("raw-data", "OUIB"), recursive = T)

       