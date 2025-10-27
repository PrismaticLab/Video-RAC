/**
 * Video-RAC Website JavaScript
 * Handles navigation, copy buttons, scroll detection, and chart rendering
 */

(function() {
    'use strict';

    // ============================================
    // Utility Functions
    // ============================================

    /**
     * Show toast notification
     */
    function showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #1D4ED8;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            z-index: 1000;
            font-size: 0.95rem;
            transition: all 0.3s ease;
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(-50%) translateY(-10px)';
            setTimeout(() => toast.remove(), 300);
        }, 2000);
    }

    /**
     * Copy text to clipboard
     */
    async function copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            showToast('Copied to clipboard!');
        } catch (err) {
            console.error('Failed to copy:', err);
            showToast('Failed to copy');
        }
    }

    // ============================================
    // Mobile Navigation Toggle
    // ============================================

    const navbarToggle = document.getElementById('navbarToggle');
    const navbarMenu = document.getElementById('navbarMenu');
    const navLinks = document.querySelectorAll('.nav-link');

    if (navbarToggle) {
        navbarToggle.addEventListener('click', () => {
            const isActive = navbarMenu.classList.contains('active');
            navbarMenu.classList.toggle('active');
            navbarToggle.setAttribute('aria-expanded', !isActive);
        });
    }

    // Close menu when clicking a nav link (mobile)
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            navbarMenu.classList.remove('active');
            navbarToggle.setAttribute('aria-expanded', 'false');
        });
    });

    // ============================================
    // Smooth Scroll for Anchor Links
    // ============================================

    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const offset = 80;
                const elementPosition = target.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - offset;
                
                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // ============================================
    // Active Section Detection on Scroll
    // ============================================

    const sections = document.querySelectorAll('section[id]');
    const navLinks2 = document.querySelectorAll('.nav-link');

    function setActiveNav() {
        let current = '';
        const scrollPosition = window.pageYOffset + 100;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');

            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                current = sectionId;
            }
        });

        navLinks2.forEach(link => {
            link.classList.remove('active');
            const href = link.getAttribute('href').substring(1);
            if (href === current) {
                link.classList.add('active');
            }
        });
    }

    window.addEventListener('scroll', setActiveNav);
    setActiveNav(); // Call once on load

    // ============================================
    // Copy to Clipboard Buttons
    // ============================================

    const copyButtons = document.querySelectorAll('.copy-btn');
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const codeBlockId = this.getAttribute('data-copy');
            const codeBlock = document.getElementById(codeBlockId);
            
            if (codeBlock) {
                // Extract text content from pre/code block
                let text = codeBlock.textContent || codeBlock.innerText;
                copyToClipboard(text.trim());
            }
        });
    });

    // ============================================
    // Collapsible Content Toggle
    // ============================================

    const collapsibleToggle = document.getElementById('schemaToggle');
    const collapsibleContent = document.getElementById('schemaContent');

    if (collapsibleToggle && collapsibleContent) {
        collapsibleToggle.addEventListener('click', function() {
            collapsibleContent.classList.toggle('active');
            collapsibleToggle.classList.toggle('active');
        });
    }

    // ============================================
    // Back to Top Button
    // ============================================

    const backToTopBtn = document.getElementById('backToTop');
    
    function toggleBackToTop() {
        if (window.pageYOffset > 300) {
            backToTopBtn.classList.add('visible');
        } else {
            backToTopBtn.classList.remove('visible');
        }
    }

    window.addEventListener('scroll', toggleBackToTop);

    if (backToTopBtn) {
        backToTopBtn.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }

    // ============================================
    // Chart Rendering (Faithfulness Comparison)
    // ============================================

    function renderChart() {
        const canvas = document.getElementById('faithfulnessChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = Math.min(canvas.width, window.innerWidth - 40);
        const height = 400;
        
        canvas.width = width;
        canvas.height = height;

        // Chart data
        const data = {
            labels: ['Image+Text', 'Text-only', 'Image-only'],
            gpt4o: [0.91, 0.85, 0.78],
            llama32: [0.88, 0.82, 0.75],
            simpleGpt4o: [0.80, 0.76, 0.72],
            simpleLlama32: [0.77, 0.73, 0.69]
        };

        const padding = { top: 40, right: 40, bottom: 60, left: 60 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;
        const barWidth = chartWidth / (data.labels.length * 4);
        const maxValue = 1.0;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw grid lines
        ctx.strokeStyle = '#E2E8F0';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 10; i++) {
            const y = padding.top + (chartHeight / 10) * i;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(width - padding.right, y);
            ctx.stroke();
        }

        // Draw axes
        ctx.strokeStyle = '#64748B';
        ctx.lineWidth = 2;
        // Y-axis
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, height - padding.bottom);
        ctx.stroke();
        // X-axis
        ctx.beginPath();
        ctx.moveTo(padding.left, height - padding.bottom);
        ctx.lineTo(width - padding.right, height - padding.bottom);
        ctx.stroke();

        // Draw Y-axis labels
        ctx.fillStyle = '#334155';
        ctx.font = '12px system-ui, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (let i = 0; i <= 10; i++) {
            const value = (maxValue / 10) * (10 - i);
            const y = padding.top + (chartHeight / 10) * i;
            ctx.fillText(value.toFixed(1), padding.left - 10, y);
        }

        // Draw bars
        const barGroups = ['gpt4o', 'llama32'];
        const colors = {
            gpt4o: '#1D4ED8',
            llama32: '#0EA5E9'
        };

        data.labels.forEach((label, labelIndex) => {
            const x = padding.left + (chartWidth / data.labels.length) * (labelIndex + 0.5);
            const labelX = x - barWidth * 0.75;

            // Draw bars for each dataset
            barGroups.forEach((dataset, datasetIndex) => {
                const value = data[dataset][labelIndex];
                const barHeight = (value / maxValue) * chartHeight;
                const barX = labelX + datasetIndex * barWidth;

                ctx.fillStyle = colors[dataset];
                ctx.fillRect(barX, height - padding.bottom, barWidth * 0.9, -barHeight);

                // Draw value on top of bar
                ctx.fillStyle = colors[dataset];
                ctx.font = 'bold 11px system-ui, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(value.toFixed(2), barX + barWidth * 0.45, height - padding.bottom - barHeight - 8);
            });
        });

        // Draw X-axis labels
        ctx.fillStyle = '#334155';
        ctx.font = '13px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        data.labels.forEach((label, labelIndex) => {
            const x = padding.left + (chartWidth / data.labels.length) * (labelIndex + 0.5);
            ctx.fillText(label, x, height - padding.bottom + 15);
        });

        // Draw legend
        const legendY = padding.top - 30;
        const legendItems = [
            { label: 'GPT-4o (Adaptive)', color: '#1D4ED8' },
            { label: 'Llama 3.2 (Adaptive)', color: '#0EA5E9' }
        ];

        let legendX = padding.left;
        legendItems.forEach(item => {
            ctx.fillStyle = item.color;
            ctx.fillRect(legendX, legendY, 16, 16);
            ctx.fillStyle = '#334155';
            ctx.font = '12px system-ui, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(item.label, legendX + 22, legendY + 2);
            legendX += item.label.length * 7 + 40;
        });
    }

    // Render chart on load
    window.addEventListener('load', () => {
        setTimeout(renderChart, 100); // Small delay to ensure canvas is ready
    });

    // Re-render chart on window resize
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(renderChart, 250);
    });

    // ============================================
    // Keyboard Navigation Support
    // ============================================

    // Add keyboard navigation for buttons
    document.querySelectorAll('button').forEach(button => {
        button.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                button.click();
            }
        });
    });

    // ============================================
    // Accessibility: Skip Link
    // ============================================

    // Add skip link if not present
    if (!document.getElementById('skip-link')) {
        const skipLink = document.createElement('a');
        skipLink.id = 'skip-link';
        skipLink.href = '#overview';
        skipLink.textContent = 'Skip to main content';
        skipLink.style.cssText = `
            position: absolute;
            left: -9999px;
            top: 0;
            z-index: 9999;
            background: #1D4ED8;
            color: white;
            padding: 12px 24px;
            border-radius: 0 0 8px 0;
        `;
        
        skipLink.addEventListener('focus', () => {
            skipLink.style.left = '0';
        });
        
        skipLink.addEventListener('blur', () => {
            skipLink.style.left = '-9999px';
        });
        
        document.body.insertBefore(skipLink, document.body.firstChild);
    }

    // ============================================
    // Initialize on Load
    // ============================================

    console.log('Video-RAC website initialized');
})();

