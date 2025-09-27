document.addEventListener('DOMContentLoaded', function () {
    // Initialize variables
    let map;
    let markers = [];
    let properties = [];
    let filteredProperties = [];

    // Navigation
    const homeLink = document.getElementById('home-link');
    const trainLink = document.getElementById('train-link');
    const uploadLink = document.getElementById('upload-link');

    const homePage = document.getElementById('home-page');
    const trainPage = document.getElementById('train-page');
    const uploadPage = document.getElementById('upload-page');

    // Home page elements
    const minPriceInput = document.getElementById('min-price');
    const maxPriceInput = document.getElementById('max-price');
    const bedsSelect = document.getElementById('beds');
    const bathsSelect = document.getElementById('baths');
    const minRoiInput = document.getElementById('min-roi');
    const gradeSelect = document.getElementById('grade');
    const neighborhoodSelect = document.getElementById('neighborhood');
    const renovationSelect = document.getElementById('renovation');
    const applyFiltersBtn = document.getElementById('apply-filters');

    const propertyList = document.getElementById('property-list');

    // Train page elements
    const trainBtn = document.getElementById('train-btn');
    const trainStatus = document.getElementById('train-status');

    // Upload page elements
    const csvFileInput = document.getElementById('csv-file');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadStatus = document.getElementById('upload-status');

    // Property modal
    const propertyModal = new bootstrap.Modal(document.getElementById('property-modal'));
    const propertyModalTitle = document.getElementById('property-modal-title');
    const propertyModalBody = document.getElementById('property-modal-body');

    // Initialize map
    function initMap() {
        map = L.map('map').setView([42.4909, -83.0167], 12); // Warren, MI coordinates

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);
    }

    // Navigation event listeners
    homeLink.addEventListener('click', function (e) {
        e.preventDefault();
        showPage('home');
    });

    trainLink.addEventListener('click', function (e) {
        e.preventDefault();
        showPage('train');
    });

    uploadLink.addEventListener('click', function (e) {
        e.preventDefault();
        showPage('upload');
    });

    // Show page function
    function showPage(page) {
        homePage.style.display = 'none';
        trainPage.style.display = 'none';
        uploadPage.style.display = 'none';

        if (page === 'home') {
            homePage.style.display = 'block';
            homeLink.classList.add('active');
            trainLink.classList.remove('active');
            uploadLink.classList.remove('active');
        } else if (page === 'train') {
            trainPage.style.display = 'block';
            homeLink.classList.remove('active');
            trainLink.classList.add('active');
            uploadLink.classList.remove('active');
        } else if (page === 'upload') {
            uploadPage.style.display = 'block';
            homeLink.classList.remove('active');
            trainLink.classList.remove('active');
            uploadLink.classList.add('active');
        }
    }

    // Load properties
    async function loadProperties() {
        try {
            // Only show loading spinner if we don't have properties yet
            if (properties.length === 0) {
                propertyList.innerHTML = '<p class="text-center"><span class="loading-spinner"></span> Loading properties...</p>';
            }

            const response = await fetch('/api/score', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to load properties');
            }

            const data = await response.json();
            
            // Log whether we got cached results
            if (data.cached) {
                console.log('Using cached properties data');
            } else {
                console.log('Fresh properties data loaded');
            }
            
            properties = data.properties;
            filteredProperties = [...properties];

            // Load filter options
            await loadFilterOptions();

            // Display properties
            displayProperties();

            // Add markers to map
            addMarkersToMap();

        } catch (error) {
            propertyList.innerHTML = `<div class="alert alert-danger">Error loading properties: ${error.message}</div>`;
            console.error('Error loading properties:', error);
        }
    }

    // Load filter options
    async function loadFilterOptions() {
        try {
            const response = await fetch('/api/filters');

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to load filter options');
            }

            const filters = await response.json();

            // Set price range
            minPriceInput.placeholder = `Min ($${filters.min_price.toLocaleString()})`;
            maxPriceInput.placeholder = `Max ($${filters.max_price.toLocaleString()})`;

            // Populate neighborhoods
            neighborhoodSelect.innerHTML = '<option value="">Any</option>';
            filters.neighborhoods.forEach(neighborhood => {
                const option = document.createElement('option');
                option.value = neighborhood;
                option.textContent = neighborhood;
                neighborhoodSelect.appendChild(option);
            });

        } catch (error) {
            console.error('Error loading filter options:', error);
        }
    }

    // Display properties
    function displayProperties() {
        propertyList.innerHTML = '';

        if (filteredProperties.length === 0) {
            propertyList.innerHTML = '<p class="text-center">No properties match your filters.</p>';
            return;
        }

        filteredProperties.forEach(property => {
            const propertyCard = document.createElement('div');
            propertyCard.className = 'card property-card';
            propertyCard.dataset.propertyId = property.property_id;

            const gradeClass = `grade-${property.grade.toLowerCase()}`;

            propertyCard.innerHTML = `
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h5 class="card-title">${property.address}</h5>
                            <p class="card-text">${property.beds} bed, ${property.baths} bath, ${property.sqft.toLocaleString()} sqft</p>
                            <p class="card-text"><strong>List Price:</strong> $${property.list_price.toLocaleString()}</p>
                            <p class="card-text"><strong>Predicted Resale:</strong> $${property.predicted_resale_value.toLocaleString()}</p>
                        </div>
                        <div class="text-center">
                            <span class="grade-badge ${gradeClass}">${property.grade}</span>
                            <p class="mt-2 mb-0"><strong>ROI:</strong> <span class="${property.roi >= 0 ? 'roi-positive' : 'roi-negative'}">${property.roi.toFixed(1)}%</span></p>
                        </div>
                    </div>
                </div>
            `;

            propertyCard.addEventListener('click', function () {
                showPropertyDetails(property);
            });

            propertyList.appendChild(propertyCard);
        });
    }

    // Add markers to map
    function addMarkersToMap() {
        // Clear existing markers
        markers.forEach(marker => {
            map.removeLayer(marker);
        });
        markers = [];

        // Add new markers
        filteredProperties.forEach(property => {
            if (property.latitude && property.longitude) {
                const gradeClass = `grade-${property.grade.toLowerCase()}`;
                const marker = L.marker([property.latitude, property.longitude]).addTo(map);

                marker.bindPopup(`
                    <div>
                        <h6>${property.address}</h6>
                        <p><strong>List Price:</strong> $${property.list_price.toLocaleString()}</p>
                        <p><strong>Grade:</strong> <span class="${gradeClass}">${property.grade}</span></p>
                        <p><strong>ROI:</strong> <span class="${property.roi >= 0 ? 'roi-positive' : 'roi-negative'}">${property.roi.toFixed(1)}%</span></p>
                        <button class="btn btn-sm btn-primary" onclick="showPropertyDetails(${JSON.stringify(property).replace(/"/g, '&quot;')})">View Details</button>
                    </div>
                `);

                markers.push(marker);
            }
        });

        // Fit map to show all markers
        if (markers.length > 0) {
            const group = new L.featureGroup(markers);
            map.fitBounds(group.getBounds().pad(0.1));
        }
    }

    // Show property details
    function showPropertyDetails(property) {
        propertyModalTitle.textContent = property.address;

        const gradeClass = `grade-${property.grade.toLowerCase()}`;
        const riskClass = property.risk_score < 33 ? 'risk-low' :
            property.risk_score < 66 ? 'risk-medium' : 'risk-high';

        propertyModalBody.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    ${property.primary_photo ? `<img src="${property.primary_photo}" class="img-fluid mb-3" alt="Property Image">` : ''}
                    <div class="property-detail-row">
                        <span class="property-detail-label">List Price:</span>
                        <span class="property-detail-value">$${property.list_price.toLocaleString()}</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Predicted Resale Value:</span>
                        <span class="property-detail-value">$${property.predicted_resale_value.toLocaleString()}</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Renovation Cost:</span>
                        <span class="property-detail-value">$${property.predicted_renovation_cost.toLocaleString()}</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Expected Profit:</span>
                        <span class="property-detail-value ${property.profit >= 0 ? 'roi-positive' : 'roi-negative'}">$${property.profit.toLocaleString()}</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">ROI:</span>
                        <span class="property-detail-value ${property.roi >= 0 ? 'roi-positive' : 'roi-negative'}">${property.roi.toFixed(1)}%</span>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="property-detail-row">
                        <span class="property-detail-label">Grade:</span>
                        <span class="property-detail-value"><span class="${gradeClass}">${property.grade}</span></span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Risk Score:</span>
                        <span class="property-detail-value ${riskClass}">${property.risk_score.toFixed(1)}/100</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Bedrooms:</span>
                        <span class="property-detail-value">${property.beds}</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Bathrooms:</span>
                        <span class="property-detail-value">${property.baths}</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Square Feet:</span>
                        <span class="property-detail-value">${property.sqft.toLocaleString()}</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Year Built:</span>
                        <span class="property-detail-value">${property.year_built}</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Renovation Level:</span>
                        <span class="property-detail-value">${property.renovation_level}</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Days on Market:</span>
                        <span class="property-detail-value">${property.days_on_mls}</span>
                    </div>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-12">
                    <h6>Investment Analysis</h6>
                    <p>${property.explanation}</p>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-12">
                    <h6>Cost Breakdown</h6>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Purchase Price:</span>
                        <span class="property-detail-value">$${property.list_price.toLocaleString()}</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Renovation Cost:</span>
                        <span class="property-detail-value">$${property.predicted_renovation_cost.toLocaleString()}</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Carrying Costs:</span>
                        <span class="property-detail-value">$${property.carrying_costs.toLocaleString()}</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Selling Costs:</span>
                        <span class="property-detail-value">$${property.selling_costs.toLocaleString()}</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Contingency:</span>
                        <span class="property-detail-value">$${property.contingency.toLocaleString()}</span>
                    </div>
                    <div class="property-detail-row">
                        <span class="property-detail-label">Total Investment:</span>
                        <span class="property-detail-value">$${property.total_costs.toLocaleString()}</span>
                    </div>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-12">
                    <a href="${property.property_url}" target="_blank" class="btn btn-primary">View on Realtor.com</a>
                </div>
            </div>
        `;

        propertyModal.show();
    }

    // Apply filters
    applyFiltersBtn.addEventListener('click', function () {
        const minPrice = minPriceInput.value ? parseFloat(minPriceInput.value) : 0;
        const maxPrice = maxPriceInput.value ? parseFloat(maxPriceInput.value) : Infinity;
        const beds = bedsSelect.value ? parseInt(bedsSelect.value) : 0;
        const baths = bathsSelect.value ? parseInt(bathsSelect.value) : 0;
        const minRoi = minRoiInput.value ? parseFloat(minRoiInput.value) : -Infinity;
        const grade = gradeSelect.value;
        const neighborhood = neighborhoodSelect.value;
        const renovation = renovationSelect.value;

        filteredProperties = properties.filter(property => {
            return property.list_price >= minPrice &&
                property.list_price <= maxPrice &&
                property.beds >= beds &&
                property.baths >= baths &&
                property.roi >= minRoi &&
                (grade === '' || property.grade === grade) &&
                (neighborhood === '' || property.neighborhood === neighborhood) &&
                (renovation === '' || property.renovation_level === renovation);
        });

        displayProperties();
        addMarkersToMap();
    });

    // Train models
    trainBtn.addEventListener('click', async function () {
        try {
            trainBtn.disabled = true;
            trainStatus.innerHTML = '<div class="status-message status-info"><span class="loading-spinner"></span> Training models...</div>';

            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to train models');
            }

            const data = await response.json();

            // Create detailed model comparison display
            let resaleComparisonHtml = '<div class="mt-3"><h6>Resale Model Comparison:</h6><div class="table-responsive"><table class="table table-sm table-striped">';
            resaleComparisonHtml += '<thead><tr><th>Model</th><th>CV RMSE</th><th>Test RMSE</th><th>Test R²</th></tr></thead><tbody>';
            
            for (const [modelName, metrics] of Object.entries(data.resale_model_performance.all_models_comparison)) {
                resaleComparisonHtml += `<tr>
                    <td>${modelName}</td>
                    <td>${metrics.cv_rmse.toFixed(2)}</td>
                    <td>${metrics.test_rmse.toFixed(2)}</td>
                    <td>${metrics.test_r2.toFixed(4)}</td>
                </tr>`;
            }
            
            resaleComparisonHtml += '</tbody></table></div></div>';
            
            // Create renovation model comparison display
            let renovationComparisonHtml = '<div class="mt-3"><h6>Renovation Model Comparison:</h6><div class="table-responsive"><table class="table table-sm table-striped">';
            renovationComparisonHtml += '<thead><tr><th>Model</th><th>CV RMSE</th><th>Test RMSE</th><th>Test R²</th></tr></thead><tbody>';
            
            for (const [modelName, metrics] of Object.entries(data.renovation_model_performance.all_models_comparison)) {
                renovationComparisonHtml += `<tr>
                    <td>${modelName}</td>
                    <td>$${metrics.cv_rmse.toFixed(2)}</td>
                    <td>$${metrics.test_rmse.toFixed(2)}</td>
                    <td>${metrics.test_r2.toFixed(4)}</td>
                </tr>`;
            }
            
            renovationComparisonHtml += '</tbody></table></div></div>';

            trainStatus.innerHTML = `
                <div class="status-message status-success">
                    <strong>Success!</strong> Models trained successfully.
                    <p><strong>Best Resale Model:</strong> ${data.resale_model_performance.best_model}</p>
                    ${resaleComparisonHtml}
                    <p><strong>Best Renovation Model:</strong> ${data.renovation_model_performance.best_model}</p>
                    ${renovationComparisonHtml}
                </div>
            `;

            // Reload properties after training
            setTimeout(loadProperties, 1000);

        } catch (error) {
            trainStatus.innerHTML = `<div class="status-message status-error">Error training models: ${error.message}</div>`;
            console.error('Error training models:', error);
        } finally {
            trainBtn.disabled = false;
        }
    });

    // Upload and analyze data
    uploadBtn.addEventListener('click', async function () {
        const file = csvFileInput.files[0];

        if (!file) {
            uploadStatus.innerHTML = '<div class="status-message status-error">Please select a file to upload.</div>';
            return;
        }

        try {
            uploadBtn.disabled = true;
            uploadStatus.innerHTML = '<div class="status-message status-info"><span class="loading-spinner"></span> Uploading and analyzing data...</div>';

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/score/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to upload and analyze data');
            }

            const data = await response.json();

            uploadStatus.innerHTML = `
                <div class="status-message status-success">
                    <strong>Success!</strong> Analyzed ${data.count} properties.
                    <br><button class="btn btn-sm btn-primary mt-2" onclick="showPage('home')">View Results</button>
                </div>
            `;

            // Update properties and switch to home page
            properties = data.properties;
            filteredProperties = [...properties];

            setTimeout(() => {
                showPage('home');
                displayProperties();
                addMarkersToMap();
            }, 1000);

        } catch (error) {
            uploadStatus.innerHTML = `<div class="status-message status-error">Error uploading and analyzing data: ${error.message}</div>`;
            console.error('Error uploading and analyzing data:', error);
        } finally {
            uploadBtn.disabled = false;
        }
    });

    // Initialize the application
    initMap();
    loadProperties();

    // Make showPropertyDetails globally accessible for popup buttons
    window.showPropertyDetails = showPropertyDetails;
    window.showPage = showPage;
});