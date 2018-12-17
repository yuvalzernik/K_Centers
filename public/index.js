var directionsService
var directionsDisplay
var directionsDisplay2
var map, places, infoWindow;
var markersRansac = [];
var markerCoreset = [];
var markerss = [];
var gmarkers = [];
var autocomplete;
var MARKER_PATH = 'https://developers.google.com/maps/documentation/javascript/images/marker_green';
var hostnameRegexp = new RegExp('^https?://.+?/');
var path = null;
var boxes;
var routeBoxer = null;
var distance = 0.3; // Distance in KM from the route where the site searches for places
var isBothOption = false

// Initialization of the map
function initMap() {
    directionsService = new google.maps.DirectionsService();
    directionsDisplay = new google.maps.DirectionsRenderer({
        suppressMarkers: true
    });
    directionsDisplay2 = new google.maps.DirectionsRenderer({
        polylineOptions: {
            strokeColor: "red",
        },suppressMarkers: true
    });
    routeBoxer = new RouteBoxer();
    map = new google.maps.Map(document.getElementById('map'), {
        zoom: 8,
        center: { lat: 37.1, lng: -95.7 },
        mapTypeControl: true,
        panControl: true,
        zoomControl: true,
        streetViewControl: true
    });
    infoWindow = new google.maps.InfoWindow({
        content: document.getElementById('info-content')
    });

    // Create the autocomplete object and associate it with the UI input control.
    var input = document.getElementById('autocomplete');
    autocomplete = new google.maps.places.Autocomplete(input);

    // Bind the map's bounds (viewport) property to the autocomplete object,
    // so that the autocomplete requests use the current map bounds for the
    // bounds option in the request.
    autocomplete.bindTo('bounds', map);

    places = new google.maps.places.PlacesService(map);

    autocomplete.addListener('place_changed', onPlaceChanged); // Add a DOM event listener to react when the user selects a place.
}

// When the user selects a city, get the place details for the city and
// zoom the map in on the city.
function onPlaceChanged() {
    var place = autocomplete.getPlace();
    if (place.geometry) {
        map.panTo(place.geometry.location);
        map.setZoom(9);
        console.log(map.getBounds())
    } else {
        document.getElementById('autocomplete').placeholder = 'Enter a city';
    }
}

// Clears all the place markers 
function clearMarkers(arrMarker) {
    console.log(arrMarker);
    for (var i = 0; i < arrMarker.length; i++) {
        if (arrMarker[i]) {
            arrMarker[i].setMap(null);
        }
    }
    arrMarker = [];
}

// Load the place information into the HTML elements used by the info window.
function buildIWContent(place) {
    document.getElementById('iw-icon').innerHTML = '<img class="hotelIcon" ' +
        'src="' + place.icon + '"/>';
    document.getElementById('iw-url').innerHTML = '<b><a href="' + place.url +
        '">' + place.name + '</a></b>';
    document.getElementById('iw-address').textContent = place.vicinity;

    if (place.formatted_phone_number) {
        document.getElementById('iw-phone-row').style.display = '';
        document.getElementById('iw-phone').textContent =
            place.formatted_phone_number;
    } else {
        document.getElementById('iw-phone-row').style.display = 'none';
    }

    if (place.rating) {
        var ratingHtml = '';
        for (var i = 0; i < 5; i++) {
            if (place.rating < (i + 0.5)) {
                ratingHtml += '&#10025;';
            } else {
                ratingHtml += '&#10029;';
            }
            document.getElementById('iw-rating-row').style.display = '';
            document.getElementById('iw-rating').innerHTML = ratingHtml;
        }
    } else {
        document.getElementById('iw-rating-row').style.display = 'none';
    }

    // The regexp isolates the first part of the URL (domain plus subdomain)
    // to give a short URL for displaying in the info window.
    if (place.website) {
        var fullUrl = place.website;
        var website = hostnameRegexp.exec(place.website);
        if (website === null) {
            website = 'http://' + place.website + '/';
            fullUrl = website;
        }
        document.getElementById('iw-website-row').style.display = '';
        document.getElementById('iw-website').textContent = website;
    } else {
        document.getElementById('iw-website-row').style.display = 'none';
    }
}

// After pressing a button, the current map bounds are saved and sent to the server
// for use of the algorithm
function getBoundsButtonFunction(RansacOrCoresetOrBoth) {
    let k_points
    let map_bounds = map.getBounds()
    let mapBoundsAndAlgorithm = {
        mapBounds: map_bounds,
        Algorithm: RansacOrCoresetOrBoth
    }
    clearMarkers(markerss)
    let myJSON = JSON.stringify(mapBoundsAndAlgorithm);
    console.log(myJSON)
    let xhr = new XMLHttpRequest();
    xhr.open('POST', '/getbounds', true);
    xhr.setRequestHeader('Content-type', 'application/json; charset=utf-8');
    xhr.onreadystatechange = function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            console.log(xhr.response)
            k_points = JSON.parse(xhr.response)
            console.log(k_points.length)
            if (k_points.length == 2) {
                isBothOption = true
                k_points.forEach(element => {
                    element.splice(0, 1)
                    element.pop()
                });
            }
            else {
                k_points.splice(0, 1)
                k_points.pop()
            }
            console.log(k_points)
            directionsDisplay.setMap(null);
            directionsDisplay2.setMap(null);
            if(isBothOption || (RansacOrCoresetOrBoth =="Ransac"))
                directionsDisplay.setMap(map);
            if(isBothOption || (RansacOrCoresetOrBoth =="Coreset"))
                directionsDisplay2.setMap(map);
            sendDirectionRequest(map, k_points, RansacOrCoresetOrBoth)

        }
    }
    xhr.send(myJSON);
}

// Searches for places around the route using Routeboxer objects
async function findPlaces(searchIndex) {
    var requestq = {
        bounds: boxes[searchIndex],
    };
    var placesTypes = []
    var checkboxes = document.querySelectorAll('input[type=checkbox]:checked')
    for (var i = 0; i < checkboxes.length; i++) {
        placesTypes.push(checkboxes[i].id)
    }
    places.nearbySearch(requestq, function (results, status) {
        if (status == google.maps.places.PlacesServiceStatus.OK) {
            for (var i = 0, result; result = results[i]; i++) {
                result.types.forEach(element => {
                    if (placesTypes.includes(element)) {
                        var marker = createMarker(result);
                    }
                });
            }
        }
        if (status != google.maps.places.PlacesServiceStatus.OVER_QUERY_LIMIT) {
            searchIndex++;
            if (searchIndex < boxes.length)
                findPlaces(searchIndex);
        } else { // delay 1 second and try again
            setTimeout("findPlaces(" + searchIndex + ")", 1000);
        }

    });
}

// Marks all the points returned by the KMeans algorithm on the map and draws
// a route between them
async function calcRoute(map, k_points, type) {
    var start;
    var end;
    var start_end = [];
    var wayp = [];
    var k_points_latlang
    k_points_latlang = Convert_points(k_points, start_end)
    for (i = 0; i < k_points_latlang.length; i++) {
        var position = k_points_latlang[i];
        createRansacCoresetMarkers(type, position)
        console.log(position)
        if (i == start_end.prototype[0]) start = position;
        else if (i == start_end.prototype[1]) end = position;
        else {
            wayp.push({
                location: position,
                stopover: true
            });
        }
    }
    var request = {
        origin: start,
        destination: end,
        travelMode: 'DRIVING',
        waypoints: wayp,
        optimizeWaypoints: true
    };
    if (type == "red") { // Coreset
        console.log("start " + start)
        console.log("end " + end)
        console.log("wayp.length " + wayp.length)
        directionsService.route(request, function (response, status) {
            if (status == 'OK') {
                directionsDisplay2.setDirections(response);
                var path = response.routes[0].overview_path;
                boxes = routeBoxer.box(path, distance);
                findPlaces(0);
            } else {
                alert("directions request failed, status=" + status)
                directionsDisplay.setMap(null);
                directionsDisplay2.setMap(null);
                clearMarkers(markerCoreset)
                clearMarkers(markersRansac)
            }
        });
    }
    else { // Ransac
        console.log("start " + start)
        console.log("end " + end)
        console.log("wayp.length " + wayp.length)
        directionsService.route(request, function (response, status) {
            if (status == 'OK') {
                directionsDisplay.setDirections(response);
                var path = response.routes[0].overview_path;
                boxes = routeBoxer.box(path, distance);
                findPlaces(0);
            } else {
                alert("directions request failed, status=" + status)
                directionsDisplay.setMap(null);
                directionsDisplay2.setMap(null);
                clearMarkers(markerCoreset)
                clearMarkers(markersRansac)
            }
        });
    }
}

// The types of the algorithms are defined by colors: Coreset = red, Ransac = blue
function sendDirectionRequest(map, k_points, type) {
    if (isBothOption) {
        clearMarkers(markerCoreset)   
        calcRoute(map, k_points[0], "red")
        clearMarkers(markersRansac)
        calcRoute(map, k_points[1], "blue")
        isBothOption = false
    }
    else {
        if(type == "Ransac"){
            if(type != "Coreset"){
                clearMarkers(markerCoreset)
            }
            clearMarkers(markersRansac)
            calcRoute(map, k_points, "blue")
        }
        else{
            if(type != "Ransac"){
                clearMarkers(markersRansac)
            }
            clearMarkers(markerCoreset)   
            calcRoute(map, k_points, "red")   
        }
    }
}

// Marks the KMeans markers on the map
function createRansacCoresetMarkers(type, pos) {
    if(type == "red")
        var url_image = "../gif/C.gif"
    else{
        var url_image = "../gif/R.gif"   
    }
    var image = {
        url: url_image,
        size: new google.maps.Size(20, 32),
        anchor: new google.maps.Point(3.5, 3.5)
    };
    var marker = new google.maps.Marker({
        map: map,
        icon: url_image,
        position: pos
    });
    if(type == "red")
        markerCoreset.push(marker);
        else{
        markersRansac.push(marker);   
        }
}

// Creates the place markers and the information windows of each one
function createMarker(place) {
    var placeLoc = place.geometry.location;
    if (place.icon) {
        var image = new google.maps.MarkerImage(
            place.icon, new google.maps.Size(71, 71),
            new google.maps.Point(0, 0), new google.maps.Point(17, 34),
            new google.maps.Size(25, 25));
    } else var image = {
        url: "https://maps.gstatic.com/intl/en_us/mapfiles/markers2/measle.png",
        size: new google.maps.Size(7, 7),
        anchor: new google.maps.Point(3.5, 3.5)
    };
    var marker = new google.maps.Marker({
        map: map,
        icon: image,
        position: place.geometry.location
    });
    markerss.push(marker);
    var request = {
        reference: place.reference
    };
    google.maps.event.addListener(marker, 'click', function () {
        places.getDetails(request, function (place, status) {
            if (status == google.maps.places.PlacesServiceStatus.OK) {
                infoWindow.open(map, marker);
                buildIWContent(place);

            } else if (status == google.maps.places.PlacesServiceStatus.OVER_QUERY_LIMIT) {
                console.log(place);
                setTimeout(function () { createMarker(place); }, 1000);
            }
            else {
                var contentStr = "<h5>No Result, status=" + status + "</h5>";
                infoWindow.setContent(contentStr);
                infoWindow.open(map, marker);
            }
        });
    });
    gmarkers.push(marker);
    if (!place.name) place.name = "result " + gmarkers.length;
    var side_bar_html = "<a href='javascript:google.maps.event.trigger(gmarkers[" + parseInt(gmarkers.length - 1) + "],\"click\");'>" + place.name + "</a><br>";
}

// Converts latitude and longitude to a LatLng objects used by Google Maps
function Convert_points(k_points, start_end) {
    var arrayOfPoints = [];
    var d = 0, maxD = 0;
    var start_index, end_index
    for (i = 0; i < k_points.length; i += 2) {
        arrayOfPoints.push(new google.maps.LatLng(k_points[i], k_points[i + 1]));
    }
    start_index = 0
    end_index = 1
    start_end.prototype = [start_index, end_index]
    return arrayOfPoints
}