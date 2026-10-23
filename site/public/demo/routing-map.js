/* Progressive enhancement of the routing map (RoutingMap.astro). Without this script the grid is a
   table of links into the text table. No network access, no storage. */
(function () {
	'use strict';
	document.querySelectorAll('[data-rm]').forEach(function (root) {
		var cells = Array.prototype.slice.call(root.querySelectorAll('.rm-cell'));
		var panel = root.querySelector('[data-panel]');
		var text = root.querySelector('.rm-text');
		if (!cells.length || !panel) return;
		panel.hidden = false;
		if (text) text.open = false;
		var selected = null;

		function show(cell) {
			var row = root.querySelector('[data-row="' + cell.getAttribute('data-cell') + '"]');
			if (!row) return;
			var d = row.querySelector('th').textContent.trim();
			var b = row.cells[1].textContent.trim();
			panel.textContent = '';
			var h = document.createElement('h4');
			h.textContent = 'D = ' + d + ', budget ' + b + ' evaluations';
			panel.appendChild(h);
			var l1 = document.createElement('p');
			l1.style.margin = '0';
			l1.textContent = 'Default stages, when no probe routes the whole budget:';
			panel.appendChild(l1);
			Array.prototype.forEach.call(row.cells[2].children, function (n) { panel.appendChild(n.cloneNode(true)); });
			var l2 = document.createElement('p');
			l2.style.margin = '0';
			l2.textContent = 'Whole-budget routes possible here:';
			panel.appendChild(l2);
			Array.prototype.forEach.call(row.cells[3].children, function (n) { panel.appendChild(n.cloneNode(true)); });
		}
		function select(cell) {
			cells.forEach(function (c) { c.removeAttribute('aria-current'); });
			cell.setAttribute('aria-current', 'true');
			selected = cell;
			show(cell);
		}
		function rowcol(cell) {
			var td = cell.parentNode, tr = td.parentNode;
			return [tr.rowIndex - 1, td.cellIndex - 1];
		}
		var grid = root.querySelector('.rm-grid');
		function cellAt(r, c) {
			var tr = grid.tBodies[0].rows[r];
			return tr && tr.cells[c + 1] ? tr.cells[c + 1].querySelector('.rm-cell') : null;
		}
		function focusable(cell) {
			cells.forEach(function (c) { c.tabIndex = -1; });
			cell.tabIndex = 0;
		}
		focusable(cells[0]);

		cells.forEach(function (cell) {
			cell.addEventListener('click', function (e) { e.preventDefault(); focusable(cell); select(cell); });
			cell.addEventListener('mouseenter', function () { show(cell); });
			cell.addEventListener('mouseleave', function () { if (selected) show(selected); else panel.innerHTML = '<p class="rm-hint">Hover, focus or tap a cell for its default stage list. Arrow keys move between cells.</p>'; });
			cell.addEventListener('focus', function () { focusable(cell); show(cell); });
			cell.addEventListener('blur', function () { if (selected) show(selected); });
			cell.addEventListener('keydown', function (e) {
				var rc = rowcol(cell), r = rc[0], c = rc[1], next = null;
				if (e.key === 'ArrowRight') next = cellAt(r, c + 1);
				else if (e.key === 'ArrowLeft') next = cellAt(r, c - 1);
				else if (e.key === 'ArrowDown') next = cellAt(r + 1, c);
				else if (e.key === 'ArrowUp') next = cellAt(r - 1, c);
				else if (e.key === 'Home') next = cellAt(r, 0);
				else if (e.key === 'End') next = cellAt(r, cells.length ? grid.tBodies[0].rows[0].cells.length - 2 : 0);
				else return;
				e.preventDefault();
				if (next) { focusable(next); next.focus(); }
			});
		});
	});
})();
