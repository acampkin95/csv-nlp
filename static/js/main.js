// Message Processor - Main JavaScript

// Utility functions
function showAlert(message, type = 'info') {
    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    $('main').prepend(alertHtml);

    // Auto-dismiss after 5 seconds
    setTimeout(function() {
        $('.alert').fadeOut('slow', function() {
            $(this).remove();
        });
    }, 5000);
}

function showLoading(containerId) {
    $(`#${containerId}`).html('<div class="spinner"></div>');
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function formatRiskLevel(level) {
    const colors = {
        'critical': 'danger',
        'high': 'warning',
        'moderate': 'info',
        'low': 'success',
        'unknown': 'secondary'
    };

    const color = colors[level.toLowerCase()] || 'secondary';
    return `<span class="badge bg-${color}">${level.toUpperCase()}</span>`;
}

// File upload handling
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const fileSize = (file.size / 1024 / 1024).toFixed(2); // MB
        console.log(`Selected file: ${file.name} (${fileSize} MB)`);

        if (fileSize > 50) {
            showAlert('File size exceeds 50MB limit', 'danger');
            event.target.value = '';
        }
    }
}

// Chart rendering helpers
function renderSentimentChart(data, containerId) {
    const trace = {
        x: data.x,
        y: data.y,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Sentiment',
        line: {
            color: '#0d6efd',
            width: 2
        },
        marker: {
            size: 6
        }
    };

    const layout = {
        title: 'Sentiment Over Time',
        xaxis: {
            title: 'Message Index'
        },
        yaxis: {
            title: 'Sentiment Score',
            range: [-1, 1]
        },
        plot_bgcolor: '#f8f9fa',
        paper_bgcolor: '#fff'
    };

    Plotly.newPlot(containerId, [trace], layout);
}

function renderRiskDistribution(data, containerId) {
    const trace = {
        labels: data.labels,
        values: data.values,
        type: 'pie',
        marker: {
            colors: ['#dc3545', '#ffc107', '#0dcaf0', '#198754']
        }
    };

    const layout = {
        title: 'Risk Level Distribution'
    };

    Plotly.newPlot(containerId, [trace], layout);
}

// AJAX error handler
$(document).ajaxError(function(event, jqxhr, settings, thrownError) {
    if (jqxhr.status === 404) {
        showAlert('Resource not found', 'warning');
    } else if (jqxhr.status === 500) {
        showAlert('Server error occurred', 'danger');
    } else if (jqxhr.status === 0) {
        showAlert('Network error. Please check your connection', 'danger');
    }
});

// Initialize tooltips and popovers
$(document).ready(function() {
    $('[data-bs-toggle="tooltip"]').tooltip();
    $('[data-bs-toggle="popover"]').popover();
});

// Export functions
window.exportToPDF = function(analysisId) {
    window.location.href = `/api/export/pdf/${analysisId}`;
};

window.exportToJSON = function(analysisId) {
    window.location.href = `/api/export/json/${analysisId}`;
};

window.exportToCSV = function(analysisId) {
    window.location.href = `/api/export/csv/${analysisId}`;
};

// Refresh functions
window.refreshCacheStats = function() {
    $.ajax({
        url: '/api/cache/stats',
        type: 'GET',
        success: function(stats) {
            console.log('Cache stats refreshed', stats);
            showAlert('Cache statistics refreshed', 'success');
        }
    });
};
