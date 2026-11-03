#pragma once

/******************************************************************************
Area of the overlap of a triangle with an axis-aligned rectangle, by
Sutherland-Hodgman clipping in double: clip the triangle against the four
sides of the rectangle in turn, then take the shoelace area of what is left.
It is the oracle for tri_cov::allocate_triangle and on purpose shares no code
with it or with the headers in include/.

The vertices are first shifted so that the lower-left corner of the rectangle
is the origin, which keeps the coordinates of a pixel's clipped piece small.
******************************************************************************/
namespace exact_clip
{
	struct Polygon
	{
		int n;
		//a triangle clipped by four half-planes has at most 7 vertices; the rounding
		//of crossing points near a corner can add up to 3 more
		double x[16], y[16];
	};

	//keep the part of p on the side s * (u - c) >= 0, where u is x (axis 0) or y (axis 1)
	inline Polygon clip(const Polygon& p, int axis, double c, double s)
	{
		Polygon q;
		q.n = 0;
		for (int i = 0; i < p.n; i++)
		{
			const int j = (i + 1) % p.n;
			const double di = s * ((axis == 0 ? p.x[i] : p.y[i]) - c);
			const double dj = s * ((axis == 0 ? p.x[j] : p.y[j]) - c);
			if (di >= 0) { q.x[q.n] = p.x[i]; q.y[q.n] = p.y[i]; q.n++; }
			if ((di < 0 && dj > 0) || (di > 0 && dj < 0))
			{
				//the crossing point, by linear interpolation; its clipped coordinate is c exactly
				const double t = di / (di - dj);
				q.x[q.n] = axis == 0 ? c : p.x[i] + t * (p.x[j] - p.x[i]);
				q.y[q.n] = axis == 1 ? c : p.y[i] + t * (p.y[j] - p.y[i]);
				q.n++;
			}
		}
		return q;
	}

	//area of triangle (x, y) inside [x0, x1] x [y0, y1]
	inline double rect_overlap(const double (&x)[3], const double (&y)[3], double x0, double y0, double x1, double y1)
	{
		Polygon p;
		p.n = 3;
		for (int k = 0; k < 3; k++) { p.x[k] = x[k] - x0; p.y[k] = y[k] - y0; }
		p = clip(p, 0, 0, 1);
		p = clip(p, 0, x1 - x0, -1);
		p = clip(p, 1, 0, 1);
		p = clip(p, 1, y1 - y0, -1);
		//shoelace about the first vertex, so that the products are of the size of the polygon
		double a = 0;
		for (int i = 1; i + 1 < p.n; i++)
			a += (p.x[i] - p.x[0]) * (p.y[i + 1] - p.y[0]) - (p.x[i + 1] - p.x[0]) * (p.y[i] - p.y[0]);
		return a < 0 ? -a / 2 : a / 2;
	}
}

//area of triangle (x, y) inside the pixel [px, px+1) x [py, py+1)
inline double exact_overlap(const double (&x)[3], const double (&y)[3], int px, int py)
{
	return exact_clip::rect_overlap(x, y, px, py, px + 1.0, py + 1.0);
}
