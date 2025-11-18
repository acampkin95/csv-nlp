/**
 * Person Manager - Complete JavaScript module for person management
 * Handles CRUD operations, real-time updates, and interactive visualizations
 */

class PersonManager {
    constructor() {
        this.persons = [];
        this.filteredPersons = [];
        this.currentPerson = null;
        this.wsConnection = null;
        this.apiBaseUrl = '/api/persons';
        this.updateInterval = null;
        this.cache = new Map();
    }

    /**
     * Initialize the person manager
     */
    init() {
        console.log('Initializing PersonManager...');

        // Load initial data
        this.loadPersons();

        // Setup event listeners
        this.setupEventListeners();

        // Initialize WebSocket connection
        this.initializeWebSocket();

        // Start auto-refresh
        this.startAutoRefresh();

        console.log('PersonManager initialized successfully');
    }

    /**
     * Setup DOM event listeners
     */
    setupEventListeners() {
        // Add person button
        $('#addPersonBtn').on('click', () => {
            this.resetPersonForm();
            $('#personModalLabel').text('Add New Person');
        });

        // Save person
        $('#savePersonBtn').on('click', () => this.savePerson());

        // Filter controls
        $('#searchInput').on('keyup', () => this.applyFilters());
        $('#riskFilter').on('change', () => this.applyFilters());
        $('#statusFilter').on('change', () => this.applyFilters());
        $('#sortBy').on('change', () => this.applyFilters());
        $('#resetFilters').on('click', () => this.resetFilters());
        $('#applyFilters').on('click', () => this.applyFilters());

        // Person card interactions
        $(document).on('click', '.person-card', (e) => {
            const personId = $(e.currentTarget).data('person-id');
            if (personId && !$(e.target).is('button') && !$(e.target).closest('button').length) {
                this.showPersonDetail(personId);
            }
        });

        // Edit person from detail modal
        $('#editPersonDetailBtn').on('click', () => this.editCurrentPerson());

        // Delete person
        $(document).on('click', '.deletePersonBtn', (e) => {
            const personId = $(e.target).closest('button').data('person-id');
            this.deletePerson(personId);
        });

        // Export person data
        $(document).on('click', '.exportPersonBtn', (e) => {
            const personId = $(e.target).closest('button').data('person-id');
            this.exportPersonData(personId);
        });
    }

    /**
     * Load all persons from API
     */
    loadPersons() {
        showLoading('personGrid');

        $.ajax({
            url: this.apiBaseUrl,
            type: 'GET',
            dataType: 'json',
            success: (data) => {
                this.persons = Array.isArray(data) ? data : data.persons || [];
                this.updateRiskDashboard();
                this.applyFilters();
                this.populatePersonFilter();
            },
            error: (xhr, status, error) => {
                console.error('Error loading persons:', error);
                showAlert('Failed to load persons', 'danger');
                $('#personGrid').html('<div class="alert alert-danger">Failed to load person data</div>');
            }
        });
    }

    /**
     * Apply filters to persons list
     */
    applyFilters() {
        const searchTerm = $('#searchInput').val().toLowerCase();
        const riskFilter = $('#riskFilter').val();
        const statusFilter = $('#statusFilter').val();
        const sortBy = $('#sortBy').val();

        this.filteredPersons = this.persons.filter(person => {
            const nameMatch = (person.name || '').toLowerCase().includes(searchTerm) ||
                            (person.email || '').toLowerCase().includes(searchTerm);
            const riskMatch = !riskFilter || (person.risk_level || '').toLowerCase() === riskFilter.toLowerCase();
            const statusMatch = !statusFilter || (person.status || '').toLowerCase() === statusFilter.toLowerCase();

            return nameMatch && riskMatch && statusMatch;
        });

        // Apply sorting
        this.sortPersons(sortBy);

        // Update count
        $('#resultCount').text(`${this.filteredPersons.length} ${this.filteredPersons.length === 1 ? 'person' : 'people'} found`);

        // Render grid
        this.renderPersonGrid();
    }

    /**
     * Sort persons by specified criteria
     */
    sortPersons(sortBy) {
        const sortFunctions = {
            'name': (a, b) => (a.name || '').localeCompare(b.name || ''),
            'risk': (a, b) => {
                const riskOrder = { critical: 0, high: 1, moderate: 2, low: 3 };
                const aRisk = riskOrder[(a.risk_level || 'low').toLowerCase()] || 3;
                const bRisk = riskOrder[(b.risk_level || 'low').toLowerCase()] || 3;
                return aRisk - bRisk;
            },
            'recent': (a, b) => new Date(b.updated_at || 0) - new Date(a.updated_at || 0),
            'interactions': (a, b) => (b.interaction_count || 0) - (a.interaction_count || 0)
        };

        const compareFn = sortFunctions[sortBy] || sortFunctions['name'];
        this.filteredPersons.sort(compareFn);
    }

    /**
     * Render person grid view
     */
    renderPersonGrid() {
        const container = $('#personGrid');

        if (this.filteredPersons.length === 0) {
            container.html(`
                <div style="grid-column: 1/-1; text-align: center; padding: 3rem;">
                    <i class="fas fa-user-slash" style="font-size: 3rem; color: #ccc;"></i>
                    <p class="text-muted mt-3">No persons found matching your filters</p>
                </div>
            `);
            return;
        }

        const html = this.filteredPersons.map(person => this.createPersonCard(person)).join('');
        container.html(html);
    }

    /**
     * Create HTML for a person card
     */
    createPersonCard(person) {
        const initials = this.getInitials(person.name || '');
        const riskClass = `risk-${(person.risk_level || 'low').toLowerCase()}`;
        const statusClass = `person-status ${(person.status || 'active').toLowerCase()}`;

        return `
            <div class="person-card" data-person-id="${person.id || ''}">
                <div class="person-card-header">
                    <div class="person-avatar">${initials}</div>
                    <div>
                        <div class="person-name">${this.escapeHtml(person.name || 'Unknown')}</div>
                        <div class="person-email">${this.escapeHtml(person.email || '')}</div>
                    </div>
                </div>
                <div class="person-card-body">
                    <div class="person-status ${(person.status || 'active').toLowerCase()}">
                        ${this.getStatusDisplay(person.status)}
                    </div>

                    <div class="person-metrics">
                        <div class="metric-box">
                            <div class="metric-value">${person.interaction_count || 0}</div>
                            <div class="metric-label">Interactions</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">${person.risk_score || 0}</div>
                            <div class="metric-label">Risk Score</div>
                        </div>
                    </div>

                    <div style="margin-bottom: 1rem;">
                        <span class="risk-badge ${riskClass}">
                            ${(person.risk_level || 'Unknown').toUpperCase()}
                        </span>
                    </div>

                    <div class="person-actions">
                        <button class="btn btn-sm btn-primary" data-bs-toggle="modal" data-bs-target="#personModal"
                                onclick="window.personManager?.editPerson('${person.id}')">
                            <i class="fas fa-edit"></i> Edit
                        </button>
                        <button class="btn btn-sm btn-info" onclick="window.personManager?.showPersonDetail('${person.id}')">
                            <i class="fas fa-eye"></i> View
                        </button>
                        <button class="btn btn-sm btn-danger deletePersonBtn" data-person-id="${person.id}">
                            <i class="fas fa-trash"></i> Delete
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Update risk dashboard statistics
     */
    updateRiskDashboard() {
        const riskCounts = {
            critical: this.persons.filter(p => (p.risk_level || '').toLowerCase() === 'critical').length,
            high: this.persons.filter(p => (p.risk_level || '').toLowerCase() === 'high').length,
            moderate: this.persons.filter(p => (p.risk_level || '').toLowerCase() === 'moderate').length,
            low: this.persons.filter(p => (p.risk_level || '').toLowerCase() === 'low').length
        };

        $('#criticalCount').text(riskCounts.critical);
        $('#highCount').text(riskCounts.high);
        $('#moderateCount').text(riskCounts.moderate);
        $('#lowCount').text(riskCounts.low);
    }

    /**
     * Show person detail modal
     */
    showPersonDetail(personId) {
        const person = this.persons.find(p => p.id === personId);
        if (!person) return;

        const detailHtml = `
            <div class="row">
                <div class="col-md-6">
                    <h6 class="text-muted">Name</h6>
                    <p>${this.escapeHtml(person.name || '')}</p>

                    <h6 class="text-muted">Email</h6>
                    <p>${this.escapeHtml(person.email || '')}</p>

                    <h6 class="text-muted">Phone</h6>
                    <p>${this.escapeHtml(person.phone || 'N/A')}</p>
                </div>
                <div class="col-md-6">
                    <h6 class="text-muted">Risk Level</h6>
                    <p><span class="risk-badge risk-${(person.risk_level || 'low').toLowerCase()}">
                        ${(person.risk_level || 'Unknown').toUpperCase()}
                    </span></p>

                    <h6 class="text-muted">Risk Score</h6>
                    <p>${person.risk_score || 0} / 100</p>

                    <h6 class="text-muted">Status</h6>
                    <p><span class="badge bg-${this.getStatusBadgeColor(person.status)}">
                        ${this.getStatusDisplay(person.status)}
                    </span></p>
                </div>
            </div>

            ${person.notes ? `
                <div class="mt-3">
                    <h6 class="text-muted">Notes</h6>
                    <p>${this.escapeHtml(person.notes)}</p>
                </div>
            ` : ''}

            ${person.tags ? `
                <div class="mt-3">
                    <h6 class="text-muted">Tags</h6>
                    <div>
                        ${person.tags.split(',').map(tag =>
                            `<span class="badge bg-secondary" style="margin-right: 0.5rem;">${this.escapeHtml(tag.trim())}</span>`
                        ).join('')}
                    </div>
                </div>
            ` : ''}

            <hr>

            <div class="row mt-3">
                <div class="col-md-4 text-center">
                    <h6 class="text-muted">Interactions</h6>
                    <p class="h4">${person.interaction_count || 0}</p>
                </div>
                <div class="col-md-4 text-center">
                    <h6 class="text-muted">Risk Assessment Notes</h6>
                    <p style="font-size: 0.9rem;">${person.risk_notes ? this.escapeHtml(person.risk_notes.substring(0, 50)) + '...' : 'None'}</p>
                </div>
                <div class="col-md-4 text-center">
                    <h6 class="text-muted">Last Updated</h6>
                    <p style="font-size: 0.9rem;">${this.formatDate(person.updated_at)}</p>
                </div>
            </div>
        `;

        $('#personDetailContent').html(detailHtml);
        $('#detailModalLabel').text(person.name || 'Person Profile');
        this.currentPerson = person;

        $('#editPersonDetailBtn').off('click').on('click', () => this.editPerson(personId));
        $('#viewInteractionsBtn').off('click').on('click', () => this.viewPersonInteractions(personId));

        const modal = new bootstrap.Modal(document.getElementById('personDetailModal'));
        modal.show();
    }

    /**
     * Edit person - load data into form
     */
    editPerson(personId) {
        const person = this.persons.find(p => p.id === personId);
        if (!person) return;

        $('#personId').val(person.id || '');
        $('#firstName').val(person.first_name || '');
        $('#lastName').val(person.last_name || '');
        $('#email').val(person.email || '');
        $('#phone').val(person.phone || '');
        $('#status').val(person.status || 'active');
        $('#riskLevel').val(person.risk_level || '');
        $('#riskScore').val(person.risk_score || '');
        $('#riskNotes').val(person.risk_notes || '');
        $('#notes').val(person.notes || '');
        $('#tags').val(person.tags || '');

        $('#personModalLabel').text('Edit Person');

        // Close detail modal if open
        const detailModal = bootstrap.Modal.getInstance(document.getElementById('personDetailModal'));
        if (detailModal) detailModal.hide();

        // Open edit modal
        const modal = new bootstrap.Modal(document.getElementById('personModal'));
        modal.show();
    }

    /**
     * Save person (create or update)
     */
    savePerson() {
        const formData = {
            id: $('#personId').val() || this.generateId(),
            name: $('#firstName').val() + ' ' + $('#lastName').val(),
            first_name: $('#firstName').val(),
            last_name: $('#lastName').val(),
            email: $('#email').val(),
            phone: $('#phone').val(),
            status: $('#status').val(),
            risk_level: $('#riskLevel').val(),
            risk_score: parseInt($('#riskScore').val()) || 0,
            risk_notes: $('#riskNotes').val(),
            notes: $('#notes').val(),
            tags: $('#tags').val()
        };

        if (!this.validatePersonForm(formData)) {
            showAlert('Please fill in all required fields', 'warning');
            return;
        }

        const isNew = !$('#personId').val();
        const method = isNew ? 'POST' : 'PUT';

        $.ajax({
            url: isNew ? this.apiBaseUrl : `${this.apiBaseUrl}/${formData.id}`,
            type: method,
            contentType: 'application/json',
            data: JSON.stringify(formData),
            success: () => {
                showAlert(`Person ${isNew ? 'created' : 'updated'} successfully`, 'success');
                const modal = bootstrap.Modal.getInstance(document.getElementById('personModal'));
                modal.hide();
                this.loadPersons();
            },
            error: (xhr, status, error) => {
                console.error('Error saving person:', error);
                showAlert('Failed to save person', 'danger');
            }
        });
    }

    /**
     * Delete person
     */
    deletePerson(personId) {
        if (!confirm('Are you sure you want to delete this person? This action cannot be undone.')) {
            return;
        }

        $.ajax({
            url: `${this.apiBaseUrl}/${personId}`,
            type: 'DELETE',
            success: () => {
                showAlert('Person deleted successfully', 'success');
                this.loadPersons();
            },
            error: (xhr, status, error) => {
                console.error('Error deleting person:', error);
                showAlert('Failed to delete person', 'danger');
            }
        });
    }

    /**
     * View person interactions
     */
    viewPersonInteractions(personId) {
        const person = this.persons.find(p => p.id === personId);
        if (!person) return;

        window.location.href = `/interactions?person_id=${personId}`;
    }

    /**
     * Reset person form
     */
    resetPersonForm() {
        $('#personId').val('');
        $('#firstName').val('');
        $('#lastName').val('');
        $('#email').val('');
        $('#phone').val('');
        $('#status').val('active');
        $('#riskLevel').val('');
        $('#riskScore').val('');
        $('#riskNotes').val('');
        $('#notes').val('');
        $('#tags').val('');
    }

    /**
     * Reset filters
     */
    resetFilters() {
        $('#searchInput').val('');
        $('#riskFilter').val('');
        $('#statusFilter').val('');
        $('#sortBy').val('name');
        this.applyFilters();
    }

    /**
     * Populate person filter dropdown
     */
    populatePersonFilter() {
        const select = $('#personFilter');
        if (select.length === 0) return; // Only on interactions page

        const options = this.persons.map(p => `<option value="${p.id}">${this.escapeHtml(p.name || '')}</option>`).join('');
        select.html('<option value="">-- Select a person --</option>' + options);
    }

    /**
     * Initialize WebSocket connection for real-time updates
     */
    initializeWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/persons`;

        try {
            this.wsConnection = new WebSocket(wsUrl);

            this.wsConnection.onopen = () => {
                console.log('WebSocket connected');
                this.updateWSStatus(true);
            };

            this.wsConnection.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWSMessage(data);
            };

            this.wsConnection.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateWSStatus(false);
                // Attempt reconnect after 3 seconds
                setTimeout(() => this.initializeWebSocket(), 3000);
            };

            this.wsConnection.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateWSStatus(false);
            };
        } catch (error) {
            console.warn('WebSocket not available:', error);
        }
    }

    /**
     * Handle WebSocket messages
     */
    handleWSMessage(data) {
        if (data.type === 'person_updated') {
            const index = this.persons.findIndex(p => p.id === data.person.id);
            if (index >= 0) {
                this.persons[index] = data.person;
            } else {
                this.persons.push(data.person);
            }
            this.updateRiskDashboard();
            this.applyFilters();
        } else if (data.type === 'person_deleted') {
            this.persons = this.persons.filter(p => p.id !== data.person_id);
            this.updateRiskDashboard();
            this.applyFilters();
        }
    }

    /**
     * Update WebSocket status indicator
     */
    updateWSStatus(connected) {
        const indicator = $('#wsStatus');
        if (connected) {
            indicator.removeClass('bg-danger').addClass('bg-success').html('<i class="fas fa-circle"></i> Connected');
        } else {
            indicator.removeClass('bg-success').addClass('bg-danger').html('<i class="fas fa-circle"></i> Disconnected');
        }
    }

    /**
     * Start auto-refresh of person data
     */
    startAutoRefresh() {
        this.updateInterval = setInterval(() => {
            this.loadPersons();
        }, 30000); // Refresh every 30 seconds
    }

    /**
     * Stop auto-refresh
     */
    stopAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }

    /**
     * Validate person form
     */
    validatePersonForm(data) {
        return data.first_name && data.last_name && data.email && data.risk_level;
    }

    /**
     * Get initials from name
     */
    getInitials(name) {
        return name.split(' ')
            .slice(0, 2)
            .map(n => n.charAt(0).toUpperCase())
            .join('');
    }

    /**
     * Get status display text
     */
    getStatusDisplay(status) {
        const displays = {
            'active': 'Active',
            'inactive': 'Inactive',
            'flagged': 'Flagged'
        };
        return displays[(status || 'active').toLowerCase()] || status;
    }

    /**
     * Get status badge color
     */
    getStatusBadgeColor(status) {
        const colors = {
            'active': 'success',
            'inactive': 'secondary',
            'flagged': 'warning'
        };
        return colors[(status || 'active').toLowerCase()] || 'secondary';
    }

    /**
     * Format date for display
     */
    formatDate(dateString) {
        if (!dateString) return 'N/A';
        try {
            const date = new Date(dateString);
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
        } catch {
            return dateString;
        }
    }

    /**
     * Generate unique ID
     */
    generateId() {
        return 'person_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    /**
     * Escape HTML special characters
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Export person data as JSON
     */
    exportPersonData(personId) {
        const person = this.persons.find(p => p.id === personId);
        if (!person) return;

        const dataStr = JSON.stringify(person, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `person_${personId}_${Date.now()}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }

    /**
     * Cleanup on page unload
     */
    destroy() {
        this.stopAutoRefresh();
        if (this.wsConnection) {
            this.wsConnection.close();
        }
    }
}

/**
 * Interaction Analyzer - Handles interaction analysis and visualization
 */
class InteractionAnalyzer {
    constructor() {
        this.interactions = [];
        this.currentPerson = null;
        this.filteredInteractions = [];
        this.wsConnection = null;
        this.apiBaseUrl = '/api/interactions';
        this.personManager = null;
    }

    /**
     * Initialize interaction analyzer
     */
    init() {
        console.log('Initializing InteractionAnalyzer...');

        // Create or get person manager instance
        if (!window.personManager) {
            window.personManager = new PersonManager();
        }
        this.personManager = window.personManager;

        // Setup event listeners
        this.setupEventListeners();

        // Load persons for filter
        this.loadPersons();

        // Initialize WebSocket
        this.initializeWebSocket();

        console.log('InteractionAnalyzer initialized successfully');
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        $('#personFilter').on('change', () => this.onPersonSelected());
        $('#interactionFilter').on('change', () => this.applyTimelineFilters());
        $('#dateFromFilter').on('change', () => this.applyTimelineFilters());
        $('#dateToFilter').on('change', () => this.applyTimelineFilters());
        $('#resetTimeline').on('click', () => this.resetTimelineFilters());
        $('#applyTimeline').on('click', () => this.applyTimelineFilters());

        // Timeline granularity controls
        $('.timeline-control-btn').on('click', (e) => {
            $('.timeline-control-btn').removeClass('active');
            $(e.target).addClass('active');
            this.renderTimeline($(e.target).data('granularity'));
        });

        // Network visualization checkboxes
        $('#showDirectConnections, #showSecondaryConnections, #showRiskNodes').on('change', () => {
            this.renderNetworkGraph();
        });

        // Intervention buttons
        $('#generateReport').on('click', () => this.generateInterventionReport());
        $('#escalateCase').on('click', () => this.escalateCase());
    }

    /**
     * Load persons for selection
     */
    loadPersons() {
        $.ajax({
            url: '/api/persons',
            type: 'GET',
            dataType: 'json',
            success: (data) => {
                this.personManager.persons = Array.isArray(data) ? data : data.persons || [];
                this.populatePersonFilter();
            }
        });
    }

    /**
     * Populate person filter
     */
    populatePersonFilter() {
        const select = $('#personFilter');
        const options = this.personManager.persons
            .map(p => `<option value="${p.id}">${this.escapeHtml(p.name || '')}</option>`)
            .join('');
        select.html('<option value="">-- Select a person --</option>' + options);
    }

    /**
     * Handle person selection
     */
    onPersonSelected() {
        const personId = $('#personFilter').val();
        if (!personId) {
            this.currentPerson = null;
            return;
        }

        this.currentPerson = this.personManager.persons.find(p => p.id === personId);
        this.loadInteractions(personId);
    }

    /**
     * Load interactions for selected person
     */
    loadInteractions(personId) {
        $.ajax({
            url: `${this.apiBaseUrl}?person_id=${personId}`,
            type: 'GET',
            dataType: 'json',
            success: (data) => {
                this.interactions = Array.isArray(data) ? data : data.interactions || [];
                this.applyTimelineFilters();
                this.updateInterventionRecommendations();
            },
            error: (xhr, status, error) => {
                console.error('Error loading interactions:', error);
                showAlert('Failed to load interactions', 'danger');
            }
        });
    }

    /**
     * Apply timeline filters
     */
    applyTimelineFilters() {
        const interactionType = $('#interactionFilter').val();
        const dateFrom = $('#dateFromFilter').val();
        const dateTo = $('#dateToFilter').val();

        this.filteredInteractions = this.interactions.filter(interaction => {
            const typeMatch = !interactionType || (interaction.type || '').toLowerCase() === interactionType.toLowerCase();
            const dateMatch = this.isDateInRange(interaction.date, dateFrom, dateTo);
            return typeMatch && dateMatch;
        });

        this.renderTimeline('day');
        this.renderTimelineEntries();
    }

    /**
     * Check if date is in range
     */
    isDateInRange(dateString, fromDate, toDate) {
        if (!dateString) return false;

        const date = new Date(dateString);
        if (fromDate) {
            const from = new Date(fromDate);
            if (date < from) return false;
        }
        if (toDate) {
            const to = new Date(toDate);
            if (date > to) return false;
        }
        return true;
    }

    /**
     * Render timeline visualization using Plotly
     */
    renderTimeline(granularity = 'day') {
        if (this.filteredInteractions.length === 0) {
            $('#timelineViz').html(`
                <div style="text-align: center; padding: 2rem; color: #6c757d;">
                    <p>No interactions found for the selected filters</p>
                </div>
            `);
            return;
        }

        // Aggregate interactions by granularity
        const aggregated = this.aggregateInteractionsByDate(this.filteredInteractions, granularity);

        const trace = {
            x: aggregated.dates,
            y: aggregated.counts,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Interactions',
            line: { color: '#0d6efd', width: 3 },
            marker: { size: 8 }
        };

        const layout = {
            title: `Interaction Timeline (${granularity})`,
            xaxis: { title: 'Date' },
            yaxis: { title: 'Number of Interactions' },
            plot_bgcolor: '#f8f9fa',
            paper_bgcolor: '#fff',
            hovermode: 'x unified'
        };

        Plotly.newPlot('timelineViz', [trace], layout, { responsive: true });
    }

    /**
     * Aggregate interactions by date
     */
    aggregateInteractionsByDate(interactions, granularity) {
        const grouped = {};

        interactions.forEach(interaction => {
            const date = new Date(interaction.date);
            let key;

            if (granularity === 'day') {
                key = date.toISOString().split('T')[0];
            } else if (granularity === 'week') {
                const weekStart = new Date(date);
                weekStart.setDate(date.getDate() - date.getDay());
                key = weekStart.toISOString().split('T')[0];
            } else if (granularity === 'month') {
                key = date.toISOString().substring(0, 7);
            }

            grouped[key] = (grouped[key] || 0) + 1;
        });

        const sortedDates = Object.keys(grouped).sort();
        return {
            dates: sortedDates,
            counts: sortedDates.map(d => grouped[d])
        };
    }

    /**
     * Render timeline entries list
     */
    renderTimelineEntries() {
        const container = $('#timelineList');

        if (this.filteredInteractions.length === 0) {
            container.html('<p class="text-muted">No interactions to display</p>');
            return;
        }

        // Sort by date descending
        const sorted = [...this.filteredInteractions].sort((a, b) =>
            new Date(b.date) - new Date(a.date)
        );

        const html = sorted.map(interaction => this.createTimelineEntry(interaction)).join('');
        container.html(html);
    }

    /**
     * Create timeline entry HTML
     */
    createTimelineEntry(interaction) {
        const riskClass = `risk-${(interaction.risk_level || 'low').toLowerCase()}`;

        return `
            <div class="timeline-entry">
                <div class="timeline-entry-date">
                    <i class="fas fa-clock"></i> ${this.formatDate(interaction.date)}
                </div>
                <div class="timeline-entry-content">
                    <strong>${interaction.type || 'Interaction'}</strong>: ${this.escapeHtml(interaction.description || '')}
                </div>
                <span class="timeline-entry-risk risk-badge ${riskClass}">
                    ${(interaction.risk_level || 'Unknown').toUpperCase()}
                </span>
            </div>
        `;
    }

    /**
     * Render network graph using D3.js
     */
    renderNetworkGraph() {
        if (!this.currentPerson) {
            $('#networkViz').html('<p class="text-muted text-center" style="padding: 2rem;">Select a person to view network</p>');
            return;
        }

        // Get network data
        const networkData = this.buildNetworkData();

        const width = $('#networkViz').width();
        const height = 500;

        // Clear previous SVG
        d3.select('#networkViz').selectAll('*').remove();

        const svg = d3.select('#networkViz')
            .attr('width', width)
            .attr('height', height);

        if (!networkData.nodes.length) {
            svg.append('text')
                .attr('x', width / 2)
                .attr('y', height / 2)
                .attr('text-anchor', 'middle')
                .attr('fill', '#999')
                .text('No network data available');
            return;
        }

        // Create force simulation
        const simulation = d3.forceSimulation(networkData.nodes)
            .force('link', d3.forceLink(networkData.links).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(30));

        // Draw links
        const link = svg.selectAll('.link')
            .data(networkData.links)
            .enter().append('line')
            .attr('class', 'link')
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', 2);

        // Draw nodes
        const node = svg.selectAll('.node')
            .data(networkData.nodes)
            .enter().append('circle')
            .attr('class', 'node')
            .attr('r', d => d.size || 20)
            .attr('fill', d => d.color)
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .call(d3.drag()
                .on('start', dragStarted)
                .on('drag', dragged)
                .on('end', dragEnded));

        // Add labels
        const text = svg.selectAll('.label')
            .data(networkData.nodes)
            .enter().append('text')
            .attr('class', 'label')
            .attr('text-anchor', 'middle')
            .attr('dy', '.3em')
            .attr('font-size', '12px')
            .text(d => d.name);

        // Update positions on simulation tick
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            text
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });

        // Drag functions
        function dragStarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragEnded(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    }

    /**
     * Build network data from interactions
     */
    buildNetworkData() {
        const nodes = [];
        const links = [];
        const nodeMap = new Map();

        // Add primary person
        const primaryNode = {
            id: this.currentPerson.id,
            name: this.currentPerson.name,
            type: 'primary',
            risk: this.currentPerson.risk_level,
            size: 30,
            color: '#667eea'
        };
        nodes.push(primaryNode);
        nodeMap.set(primaryNode.id, 0);

        // Add connected persons from interactions
        const connectedIds = new Set();
        this.interactions.forEach(interaction => {
            if (interaction.related_person_id && !connectedIds.has(interaction.related_person_id)) {
                connectedIds.add(interaction.related_person_id);

                // Find person details
                const relatedPerson = this.personManager.persons.find(p => p.id === interaction.related_person_id);
                if (relatedPerson) {
                    const node = {
                        id: relatedPerson.id,
                        name: relatedPerson.name,
                        type: 'connected',
                        risk: relatedPerson.risk_level,
                        size: 20,
                        color: this.getRiskColor(relatedPerson.risk_level)
                    };

                    // Filter by risk if checkbox is selected
                    if ($('#showRiskNodes').is(':checked') && relatedPerson.risk_level !== 'critical' && relatedPerson.risk_level !== 'high') {
                        return;
                    }

                    nodes.push(node);
                    nodeMap.set(node.id, nodes.length - 1);

                    // Add link
                    links.push({
                        source: this.currentPerson.id,
                        target: relatedPerson.id,
                        value: this.interactions.filter(i => i.related_person_id === relatedPerson.id).length
                    });
                }
            }
        });

        return { nodes, links };
    }

    /**
     * Get color for risk level
     */
    getRiskColor(riskLevel) {
        const colors = {
            critical: '#dc3545',
            high: '#ffc107',
            moderate: '#0dcaf0',
            low: '#198754'
        };
        return colors[(riskLevel || 'low').toLowerCase()] || '#6c757d';
    }

    /**
     * Render risk progression chart
     */
    renderRiskProgression() {
        if (!this.currentPerson) return;

        // Get risk progression data
        const progressionData = this.buildRiskProgressionData();

        const trace = {
            x: progressionData.dates,
            y: progressionData.scores,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Risk Score',
            line: { color: '#dc3545', width: 3 },
            marker: { size: 8, color: '#dc3545' },
            fill: 'tozeroy',
            fillcolor: 'rgba(220, 53, 69, 0.1)'
        };

        const layout = {
            title: 'Risk Score Progression',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Risk Score (0-100)', range: [0, 100] },
            plot_bgcolor: '#f8f9fa',
            paper_bgcolor: '#fff',
            hovermode: 'x unified'
        };

        Plotly.newPlot('riskProgressionChart', [trace], layout, { responsive: true });
    }

    /**
     * Build risk progression data
     */
    buildRiskProgressionData() {
        // This would typically come from the backend
        // For now, create mock data showing progression
        const dates = [];
        const scores = [];

        for (let i = 0; i < 30; i++) {
            const date = new Date();
            date.setDate(date.getDate() - (30 - i));
            dates.push(date.toISOString().split('T')[0]);

            // Mock risk score progression
            const baseScore = parseInt(this.currentPerson.risk_score) || 50;
            const variation = Math.sin(i / 5) * 10 + Math.random() * 5;
            scores.push(Math.min(100, Math.max(0, baseScore + variation)));
        }

        return { dates, scores };
    }

    /**
     * Update intervention recommendations
     */
    updateInterventionRecommendations() {
        // Build recommendations based on risk level and interactions
        const recommendations = this.buildInterventionRecommendations();

        const html = recommendations.map(rec => this.createInterventionItem(rec)).join('');
        $('#interventionsList').html(html || '<p class="text-muted">No interventions recommended</p>');

        // Update summary counts
        const urgent = recommendations.filter(r => r.priority === 'urgent').length;
        const important = recommendations.filter(r => r.priority === 'important').length;
        const routine = recommendations.filter(r => r.priority === 'routine').length;

        $('#urgentCount').text(urgent);
        $('#importantCount').text(important);
        $('#routineCount').text(routine);
    }

    /**
     * Build intervention recommendations
     */
    buildInterventionRecommendations() {
        const recommendations = [];

        if (!this.currentPerson) return recommendations;

        const riskLevel = (this.currentPerson.risk_level || 'low').toLowerCase();

        if (riskLevel === 'critical') {
            recommendations.push({
                title: 'Immediate Safety Assessment Required',
                description: 'Person classified as critical risk. Schedule immediate safety assessment with qualified personnel.',
                priority: 'urgent'
            });

            recommendations.push({
                title: 'Escalate to Senior Management',
                description: 'Critical risk cases must be escalated to senior management for decision-making.',
                priority: 'urgent'
            });
        }

        if (riskLevel === 'critical' || riskLevel === 'high') {
            recommendations.push({
                title: 'Increase Monitoring Frequency',
                description: 'Implement daily monitoring and interaction review for high-risk individuals.',
                priority: 'urgent'
            });

            recommendations.push({
                title: 'Document All Interactions',
                description: 'Ensure all interactions are properly documented with timestamps and context.',
                priority: 'important'
            });
        }

        if (riskLevel === 'moderate') {
            recommendations.push({
                title: 'Schedule Review Meeting',
                description: 'Schedule a risk assessment review meeting within the next 7 days.',
                priority: 'important'
            });
        }

        // Add interaction-based recommendations
        if (this.filteredInteractions.length > 20) {
            recommendations.push({
                title: 'High Interaction Volume',
                description: `${this.filteredInteractions.length} interactions detected. Review interaction patterns for concerning behaviors.`,
                priority: 'important'
            });
        }

        // Generic recommendation
        if (recommendations.length === 0) {
            recommendations.push({
                title: 'Continue Routine Monitoring',
                description: 'Person poses low risk. Continue routine monitoring and documentation.',
                priority: 'routine'
            });
        }

        return recommendations;
    }

    /**
     * Create intervention item HTML
     */
    createInterventionItem(intervention) {
        return `
            <div class="intervention-item ${intervention.priority}">
                <div style="flex-grow: 1;">
                    <div class="intervention-title">${intervention.title}</div>
                    <div class="intervention-description">${intervention.description}</div>
                </div>
                <span class="intervention-priority ${intervention.priority}">
                    ${intervention.priority.toUpperCase()}
                </span>
            </div>
        `;
    }

    /**
     * Generate intervention report
     */
    generateInterventionReport() {
        if (!this.currentPerson) {
            showAlert('Please select a person first', 'warning');
            return;
        }

        showAlert('Generating intervention report...', 'info');

        $.ajax({
            url: `/api/reports/intervention/${this.currentPerson.id}`,
            type: 'POST',
            dataType: 'json',
            success: (data) => {
                showAlert('Report generated successfully', 'success');
                // Trigger download
                const link = document.createElement('a');
                link.href = data.file_url;
                link.download = `intervention_report_${this.currentPerson.id}.pdf`;
                link.click();
            },
            error: () => {
                showAlert('Failed to generate report', 'danger');
            }
        });
    }

    /**
     * Escalate case
     */
    escalateCase() {
        if (!this.currentPerson) {
            showAlert('Please select a person first', 'warning');
            return;
        }

        $.ajax({
            url: `/api/cases/escalate/${this.currentPerson.id}`,
            type: 'POST',
            dataType: 'json',
            success: () => {
                showAlert('Case escalated successfully', 'success');
            },
            error: () => {
                showAlert('Failed to escalate case', 'danger');
            }
        });
    }

    /**
     * Reset timeline filters
     */
    resetTimelineFilters() {
        $('#interactionFilter').val('');
        $('#dateFromFilter').val('');
        $('#dateToFilter').val('');
        this.applyTimelineFilters();
    }

    /**
     * Format date for display
     */
    formatDate(dateString) {
        if (!dateString) return 'N/A';
        try {
            const date = new Date(dateString);
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
        } catch {
            return dateString;
        }
    }

    /**
     * Escape HTML special characters
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Initialize WebSocket for real-time updates
     */
    initializeWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/interactions`;

        try {
            this.wsConnection = new WebSocket(wsUrl);

            this.wsConnection.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'new_interaction' && this.currentPerson && data.person_id === this.currentPerson.id) {
                    this.interactions.push(data.interaction);
                    this.applyTimelineFilters();
                }
            };
        } catch (error) {
            console.warn('WebSocket not available:', error);
        }
    }
}

// Make classes available globally
window.PersonManager = PersonManager;
window.InteractionAnalyzer = InteractionAnalyzer;
