let userLat = 12.879376;
let userLng = 77.544295;

// user_lat = 12.879376
// user_lng = 77.544295


function fetchNearbyDoctors() {
    if (!navigator.geolocation) {
        alert("Geolocation not supported");
        return;
    }

    navigator.geolocation.getCurrentPosition(position => {
        userLat = position.coords.latitude;
        userLng = position.coords.longitude;

        fetch('/nearby-doctors', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat: 12.879376, lng: 77.544295 })
        })
        .then(res => res.json())
        .then(doctors => {
            displayDoctors(doctors);
        })
        .catch(err => {
            console.error(err);
            alert("Failed to fetch doctors.");
        });
    });
}

function displayDoctors(doctors) {
    const list = document.getElementById("doctorList");
    list.innerHTML = "";

    doctors.forEach((doc, index) => {
        const li = document.createElement("li");
        li.innerHTML = `
            <strong>${doc.name}</strong><br>
            ${doc.address}<br>
            ${doc.distance}<br>
            <button onclick="showDirections(${doc.lat}, ${doc.lng})">Get Directions</button>
        `;
        list.appendChild(li);
    });
}

function showDirections(destLat, destLng) {
    const directionsService = new google.maps.DirectionsService();
    const directionsRenderer = new google.maps.DirectionsRenderer();
    const map = new google.maps.Map(document.getElementById('map'), {
        zoom: 14,
        center: { lat: userLat, lng: userLng }
    });
    directionsRenderer.setMap(map);

    const request = {
        origin: { lat: userLat, lng: userLng },
        destination: { lat: destLat, lng: destLng },
        travelMode: google.maps.TravelMode.DRIVING
    };

    directionsService.route(request, function(result, status) {
        if (status === 'OK') {
            directionsRenderer.setDirections(result);
        } else {
            alert("Directions request failed: " + status);
        }
    });
}
